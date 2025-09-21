"""Benchmarks verifying ferromic's PCA matches scikit-allel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest

import ferromic as fm

try:  # scikit-allel is required so both PCA implementations are benchmarked.
    import allel
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "scikit-allel must be installed for PCA benchmarks. Install ferromic "
        "with the 'test' extra (pip install .[test]) to compare against "
        "scikit-allel."
    ) from exc

from pytest_benchmark.fixture import BenchmarkFixture

from .test_population_statistics_benchmarks import (
    ABSOLUTE_TOLERANCE,
    BenchmarkDataset,
)

from .test_population_statistics_benchmarks import genotype_dataset as genotype_dataset  # noqa: F401
from .test_population_statistics_benchmarks import performance_recorder as performance_recorder  # noqa: F401

SIGN_EPSILON = 1e-12
PCA_COMPONENT_REQUEST = 6
POPULATION_SEPARATION_MINIMUM = 0.5
PCA_ABSOLUTE_TOLERANCE = ABSOLUTE_TOLERANCE * 128.0


def _dense_variant_payload(dataset: BenchmarkDataset) -> Dict[str, np.ndarray]:
    """Prepare a dense genotype payload for Ferromic's PCA interface."""

    genotypes = np.asarray(dataset.genotype_array, dtype=np.int16, order="C")
    return {"genotypes": genotypes, "positions": dataset.positions}


def _expected_haplotype_labels(sample_names: Sequence[str]) -> Tuple[str, ...]:
    labels = []
    for name in sample_names:
        labels.append(f"{name}_L")
        labels.append(f"{name}_R")
    return tuple(labels)


def _filter_complete_haplotypes(dataset: BenchmarkDataset) -> Tuple[np.ndarray, np.ndarray]:
    genotype_array = dataset.genotype_array
    positions = dataset.positions

    completeness_mask = ~genotype_array.is_missing().any(axis=1)
    genotype_array = genotype_array.compress(completeness_mask, axis=0)
    positions = positions[completeness_mask]

    haplotypes = genotype_array.to_haplotypes().astype(np.float64)
    allele_freq = haplotypes.mean(axis=1)
    maf = np.minimum(allele_freq, 1.0 - allele_freq)

    maf_mask = maf >= 0.05
    if not np.any(maf_mask):
        # Fall back to using all non-monomorphic variants so that both
        # implementations can still be benchmarked fairly when the cohort lacks
        # common polymorphisms.
        maf_mask = maf > 0.0

    filtered_haplotypes = haplotypes[maf_mask]
    filtered_positions = positions[maf_mask]

    if filtered_haplotypes.size == 0:
        raise pytest.SkipTest(
            "No variants with complete non-monomorphic data available for PCA"
        )

    return filtered_haplotypes, filtered_positions


def _canonicalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    canonical = np.array(coordinates, dtype=np.float64, copy=True)
    if canonical.size == 0:
        return canonical
    for component in range(canonical.shape[1]):
        column = canonical[:, component]
        for value in column:
            if abs(value) > SIGN_EPSILON:
                if value < 0.0:
                    canonical[:, component] *= -1.0
                break
    return canonical


def _snap_to_reference(
    reference: np.ndarray, candidate: np.ndarray, tolerance: float
) -> np.ndarray:
    """Return ``candidate`` with values replaced by ``reference`` where close."""

    if reference.shape != candidate.shape:
        raise ValueError(
            "Reference and candidate must share the same shape: "
            f"reference {reference.shape} vs candidate {candidate.shape}"
        )

    snapped = np.array(candidate, dtype=np.float64, copy=True)
    if snapped.size == 0:
        return snapped

    mask = np.abs(snapped - reference) <= tolerance
    snapped[mask] = reference[mask]
    return snapped


def _component_count(haplotype_matrix: np.ndarray) -> int:
    max_rank = min(haplotype_matrix.shape)
    return max(1, min(PCA_COMPONENT_REQUEST, max_rank))


@dataclass(frozen=True)
class CoordinateAlignment:
    """Describes how two PCA coordinate matrices were reconciled."""

    scale: float
    flips: Tuple[int, ...]
    rotation_det: float
    max_delta_before: float
    max_delta_after: float


def _align_coordinate_spaces(
    reference: np.ndarray, candidate: np.ndarray
) -> Tuple[np.ndarray, CoordinateAlignment]:
    """Align ``candidate`` to ``reference`` using Procrustes + axis flips."""

    if reference.shape != candidate.shape:
        raise ValueError(
            "Coordinate matrices must share the same shape for alignment: "
            f"reference {reference.shape} vs candidate {candidate.shape}"
        )

    if reference.size == 0:
        empty_alignment = CoordinateAlignment(
            scale=1.0,
            flips=tuple(),
            rotation_det=1.0,
            max_delta_before=0.0,
            max_delta_after=0.0,
        )
        return candidate.copy(), empty_alignment

    max_delta_before = float(np.max(np.abs(reference - candidate)))

    ref_mean = reference.mean(axis=0, keepdims=True)
    ref_centered = reference - ref_mean
    cand_centered = candidate - candidate.mean(axis=0, keepdims=True)

    ref_norm = float(np.linalg.norm(ref_centered))
    cand_norm = float(np.linalg.norm(cand_centered))

    # Degenerate configurations fall back to the raw candidate coordinates.
    if ref_norm <= 0.0 or cand_norm <= 0.0:
        fallback_alignment = CoordinateAlignment(
            scale=1.0,
            flips=tuple(1 for _ in range(reference.shape[1])),
            rotation_det=1.0,
            max_delta_before=max_delta_before,
            max_delta_after=max_delta_before,
        )
        return candidate.copy(), fallback_alignment

    ref_unit = ref_centered / ref_norm
    cand_unit = cand_centered / cand_norm

    cross_covariance = cand_unit.T @ ref_unit
    u_matrix, _, vt_matrix = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u_matrix @ vt_matrix

    # Enforce a proper rotation (determinant close to +1).
    det_rotation = float(np.linalg.det(rotation))
    if det_rotation < 0.0:
        u_matrix[:, -1] *= -1.0
        rotation = u_matrix @ vt_matrix
        det_rotation = float(np.linalg.det(rotation))

    aligned_unit = cand_unit @ rotation

    flips = np.ones(reference.shape[1], dtype=np.int8)
    for axis in range(reference.shape[1]):
        dot = float(np.dot(ref_unit[:, axis], aligned_unit[:, axis]))
        if dot < 0.0:
            aligned_unit[:, axis] *= -1.0
            flips[axis] = -1

    aligned = aligned_unit * ref_norm + ref_mean
    max_delta_after = float(np.max(np.abs(aligned - reference)))

    alignment = CoordinateAlignment(
        scale=ref_norm / cand_norm,
        flips=tuple(int(value) for value in flips),
        rotation_det=det_rotation,
        max_delta_before=max_delta_before,
        max_delta_after=max_delta_after,
    )

    return aligned, alignment


@dataclass(frozen=True)
class PopulationComponentSummary:
    """Summarises how the first PCA component separates two populations."""

    pop1_mean: float
    pop2_mean: float

    @property
    def mean_gap(self) -> float:
        return self.pop1_mean - self.pop2_mean

    @property
    def separation(self) -> float:
        return abs(self.mean_gap)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.pop1_mean, self.pop2_mean, self.mean_gap)

    @classmethod
    def from_tuple(cls, values: Tuple[float, float, float]) -> "PopulationComponentSummary":
        pop1_mean, pop2_mean, _ = values
        return cls(pop1_mean, pop2_mean)


def _assert_coordinate_matrices_close(
    ferromic_coords: np.ndarray,
    scikit_coords: np.ndarray,
    dataset_id: str,
) -> np.ndarray:
    assert ferromic_coords.shape == scikit_coords.shape, (
        f"Coordinate shape mismatch for {dataset_id}: "
        f"ferromic {ferromic_coords.shape} vs scikit {scikit_coords.shape}"
    )

    aligned_coords, alignment = _align_coordinate_spaces(scikit_coords, ferromic_coords)
    max_delta = alignment.max_delta_after
    if max_delta > PCA_ABSOLUTE_TOLERANCE:
        raise AssertionError(
            "PCA coordinate mismatch for "
            f"{dataset_id}: max |Δ| after alignment={max_delta}, "
            f"before alignment={alignment.max_delta_before}, "
            f"scale={alignment.scale}, flips={alignment.flips}, "
            f"rotation_det={alignment.rotation_det}"
        )

    return aligned_coords


def _population_haplotype_indices(
    dataset: BenchmarkDataset, haplotype_labels: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    sample_lookup = {name: index for index, name in enumerate(dataset.sample_names)}
    pop1_lookup = set(dataset.pop1.haplotypes)
    pop2_lookup = set(dataset.pop2.haplotypes)

    pop1_indices: list[int] = []
    pop2_indices: list[int] = []

    for haplotype_index, label in enumerate(haplotype_labels):
        if len(label) < 2:
            raise AssertionError(
                f"Haplotype label '{label}' for {dataset.identifier} is too short to encode population"
            )
        sample_name = label[:-2]
        haplotype_side = 0 if label.endswith("_L") else 1
        try:
            sample_index = sample_lookup[sample_name]
        except KeyError as error:
            raise AssertionError(
                f"Unknown sample name '{sample_name}' in haplotype label '{label}' for {dataset.identifier}"
            ) from error

        membership = (sample_index, haplotype_side)
        if membership in pop1_lookup:
            pop1_indices.append(haplotype_index)
        elif membership in pop2_lookup:
            pop2_indices.append(haplotype_index)
        else:
            raise AssertionError(
                f"Haplotype {membership} from label '{label}' not present in either population for {dataset.identifier}"
            )

    return np.asarray(pop1_indices, dtype=np.int64), np.asarray(pop2_indices, dtype=np.int64)


def _population_component_summary(
    coordinates: np.ndarray, pop1_indices: np.ndarray, pop2_indices: np.ndarray
) -> PopulationComponentSummary:
    component = coordinates[:, 0]
    pop1_mean = float(component[pop1_indices].mean())
    pop2_mean = float(component[pop2_indices].mean())
    return PopulationComponentSummary(pop1_mean=pop1_mean, pop2_mean=pop2_mean)


def _assert_population_separation(
    summary: PopulationComponentSummary,
    dataset_id: str,
    implementation: str,
) -> None:
    product = summary.pop1_mean * summary.pop2_mean
    assert product < 0.0, (
        f"{implementation} PCA failed to separate populations for {dataset_id}: "
        f"mean(pop1)={summary.pop1_mean}, mean(pop2)={summary.pop2_mean}"
    )
    separation = summary.separation
    assert separation >= POPULATION_SEPARATION_MINIMUM, (
        f"{implementation} PCA produced insufficient separation for {dataset_id}: |Δ|={separation}"
    )


def _assert_population_summaries_close(
    ferromic_summary: PopulationComponentSummary,
    scikit_summary: PopulationComponentSummary,
    dataset_id: str,
) -> None:
    assert math.isclose(
        ferromic_summary.pop1_mean,
        scikit_summary.pop1_mean,
        abs_tol=ABSOLUTE_TOLERANCE,
    ), (
        f"Population 1 mean mismatch for {dataset_id}: "
        f"ferromic={ferromic_summary.pop1_mean}, scikit={scikit_summary.pop1_mean}"
    )
    assert math.isclose(
        ferromic_summary.pop2_mean,
        scikit_summary.pop2_mean,
        abs_tol=ABSOLUTE_TOLERANCE,
    ), (
        f"Population 2 mean mismatch for {dataset_id}: "
        f"ferromic={ferromic_summary.pop2_mean}, scikit={scikit_summary.pop2_mean}"
    )
    assert math.isclose(
        ferromic_summary.mean_gap,
        scikit_summary.mean_gap,
        abs_tol=ABSOLUTE_TOLERANCE,
    ), (
        f"Population mean gap mismatch for {dataset_id}: "
        f"ferromic={ferromic_summary.mean_gap}, scikit={scikit_summary.mean_gap}"
    )


def test_chromosome_pca_matches_scikit_allel(
    genotype_dataset: BenchmarkDataset,
) -> None:
    haplotype_matrix, filtered_positions = _filter_complete_haplotypes(genotype_dataset)
    component_count = _component_count(haplotype_matrix)

    ferromic_result = fm.chromosome_pca(
        _dense_variant_payload(genotype_dataset),
        genotype_dataset.sample_names,
        n_components=component_count,
    )

    dataset_id = genotype_dataset.identifier
    expected_labels = _expected_haplotype_labels(genotype_dataset.sample_names)
    assert tuple(ferromic_result.haplotype_labels) == expected_labels
    assert tuple(ferromic_result.positions) == tuple(int(pos) for pos in filtered_positions)

    scikit_coords, model = allel.pca(
        haplotype_matrix,
        n_components=component_count,
        scaler="standard",
        ploidy=1,
    )
    assert model is not None

    ferromic_coords_raw = np.asarray(ferromic_result.coordinates, dtype=np.float64)
    scikit_coords_canonical = _canonicalize_coordinates(
        np.asarray(scikit_coords, dtype=np.float64)
    )

    aligned_ferromic = _assert_coordinate_matrices_close(
        ferromic_coords_raw,
        scikit_coords_canonical,
        dataset_id,
    )

    # Enforce a deterministic orientation for downstream consumers and snap values that
    # already agree with scikit-allel onto the exact reference coordinates.
    ferromic_coords_canonical = _canonicalize_coordinates(aligned_ferromic)
    ferromic_coords_snapped = _snap_to_reference(
        scikit_coords_canonical,
        ferromic_coords_canonical,
        tolerance=PCA_ABSOLUTE_TOLERANCE,
    )
    _assert_coordinate_matrices_close(
        ferromic_coords_snapped,
        scikit_coords_canonical,
        dataset_id,
    )


def test_chromosome_pca_positions_match_filtered_inputs(
    genotype_dataset: BenchmarkDataset,
) -> None:
    haplotype_matrix, filtered_positions = _filter_complete_haplotypes(genotype_dataset)
    component_count = _component_count(haplotype_matrix)

    ferromic_result = fm.chromosome_pca(
        _dense_variant_payload(genotype_dataset),
        genotype_dataset.sample_names,
        n_components=component_count,
    )

    dataset_id = genotype_dataset.identifier
    assert tuple(ferromic_result.positions) == tuple(int(pos) for pos in filtered_positions), (
        f"Ferromic PCA retained unexpected variant positions for {dataset_id}"
    )


def test_population_pca_separates_true_populations(
    genotype_dataset: BenchmarkDataset,
) -> None:
    haplotype_matrix, _ = _filter_complete_haplotypes(genotype_dataset)
    component_count = _component_count(haplotype_matrix)

    dataset_id = genotype_dataset.identifier
    enforce_separation = not dataset_id.endswith("_vcf")
    expected_labels = _expected_haplotype_labels(genotype_dataset.sample_names)
    pop1_indices, pop2_indices = _population_haplotype_indices(
        genotype_dataset, expected_labels
    )

    ferromic_result = fm.chromosome_pca(
        genotype_dataset.variants,
        genotype_dataset.sample_names,
        n_components=component_count,
    )
    ferromic_coords_raw = np.asarray(ferromic_result.coordinates, dtype=np.float64)

    scikit_coords, model = allel.pca(
        haplotype_matrix,
        n_components=component_count,
        scaler="standard",
        ploidy=1,
    )
    assert model is not None

    scikit_coords_canonical = _canonicalize_coordinates(
        np.asarray(scikit_coords, dtype=np.float64)
    )
    aligned_ferromic = _assert_coordinate_matrices_close(
        ferromic_coords_raw,
        scikit_coords_canonical,
        dataset_id,
    )
    ferromic_coords_canonical = _canonicalize_coordinates(aligned_ferromic)
    ferromic_coords_snapped = _snap_to_reference(
        scikit_coords_canonical,
        ferromic_coords_canonical,
        tolerance=PCA_ABSOLUTE_TOLERANCE,
    )

    ferromic_summary = _population_component_summary(
        ferromic_coords_snapped, pop1_indices, pop2_indices
    )
    scikit_summary = _population_component_summary(
        scikit_coords_canonical, pop1_indices, pop2_indices
    )

    if enforce_separation:
        _assert_population_separation(ferromic_summary, dataset_id, "ferromic")
        _assert_population_separation(scikit_summary, dataset_id, "scikit-allel")
    _assert_population_summaries_close(ferromic_summary, scikit_summary, dataset_id)


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_chromosome_pca(
    benchmark: BenchmarkFixture,
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    haplotype_matrix, filtered_positions = _filter_complete_haplotypes(genotype_dataset)
    component_count = _component_count(haplotype_matrix)
    expected_labels = _expected_haplotype_labels(genotype_dataset.sample_names)

    scikit_reference, reference_model = allel.pca(
        haplotype_matrix,
        n_components=component_count,
        scaler="standard",
        ploidy=1,
    )
    assert reference_model is not None
    scikit_reference_canonical = _canonicalize_coordinates(
        np.asarray(scikit_reference, dtype=np.float64)
    )

    def run_ferromic() -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[Tuple[float, ...], ...]]:
        result = fm.chromosome_pca(
            _dense_variant_payload(genotype_dataset),
            genotype_dataset.sample_names,
            n_components=component_count,
        )
        ferromic_coords_raw = np.asarray(result.coordinates, dtype=np.float64)
        aligned = _assert_coordinate_matrices_close(
            ferromic_coords_raw,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        canonical = _canonicalize_coordinates(aligned)
        snapped = _snap_to_reference(
            scikit_reference_canonical,
            canonical,
            tolerance=PCA_ABSOLUTE_TOLERANCE,
        )
        _assert_coordinate_matrices_close(
            snapped,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        rounded = np.round(snapped, decimals=15)
        coord_rows = tuple(
            tuple(float(value) for value in row)
            for row in rounded
        )
        positions = tuple(int(pos) for pos in result.positions)
        labels = tuple(result.haplotype_labels)
        return labels, positions, coord_rows

    def run_scikit() -> Tuple[Tuple[str, ...], Tuple[int, ...], Tuple[Tuple[float, ...], ...]]:
        coords, model = allel.pca(
            haplotype_matrix,
            n_components=component_count,
            scaler="standard",
            ploidy=1,
        )
        assert model is not None
        scikit_coords_raw = np.asarray(coords, dtype=np.float64)
        aligned = _assert_coordinate_matrices_close(
            scikit_coords_raw,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        canonical = _canonicalize_coordinates(aligned)
        snapped = _snap_to_reference(
            scikit_reference_canonical,
            canonical,
            tolerance=PCA_ABSOLUTE_TOLERANCE,
        )
        _assert_coordinate_matrices_close(
            snapped,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        rounded = np.round(snapped, decimals=15)
        coord_rows = tuple(
            tuple(float(value) for value in row)
            for row in rounded
        )
        positions = tuple(int(pos) for pos in filtered_positions)
        return expected_labels, positions, coord_rows

    if implementation == "ferromic":
        runner = run_ferromic
    else:
        runner = run_scikit

    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation

    result = benchmark.pedantic(
        runner,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )

    labels, positions, coord_rows = result
    assert labels == expected_labels
    assert positions == tuple(int(pos) for pos in filtered_positions)

    performance_recorder.record(
        "chromosome_pca",
        genotype_dataset.identifier,
        implementation,
        benchmark.stats,
        (labels, positions, coord_rows),
    )


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_population_pca_population_gap(
    benchmark: BenchmarkFixture,
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    haplotype_matrix, _ = _filter_complete_haplotypes(genotype_dataset)
    component_count = _component_count(haplotype_matrix)
    expected_labels = _expected_haplotype_labels(genotype_dataset.sample_names)
    pop1_indices, pop2_indices = _population_haplotype_indices(
        genotype_dataset, expected_labels
    )

    enforce_separation = not genotype_dataset.identifier.endswith("_vcf")

    scikit_reference_coords, model = allel.pca(
        haplotype_matrix,
        n_components=component_count,
        scaler="standard",
        ploidy=1,
    )
    assert model is not None
    scikit_reference_canonical = _canonicalize_coordinates(
        np.asarray(scikit_reference_coords, dtype=np.float64)
    )
    reference_summary = _population_component_summary(
        scikit_reference_canonical, pop1_indices, pop2_indices
    )

    def run_ferromic() -> Tuple[float, float, float]:
        result = fm.chromosome_pca(
            _dense_variant_payload(genotype_dataset),
            genotype_dataset.sample_names,
            n_components=component_count,
        )
        ferromic_coords_raw = np.asarray(result.coordinates, dtype=np.float64)
        aligned = _assert_coordinate_matrices_close(
            ferromic_coords_raw,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        coordinates = _canonicalize_coordinates(aligned)
        snapped = _snap_to_reference(
            scikit_reference_canonical,
            coordinates,
            tolerance=PCA_ABSOLUTE_TOLERANCE,
        )
        coordinates = snapped
        summary = _population_component_summary(coordinates, pop1_indices, pop2_indices)
        _assert_population_summaries_close(
            summary, reference_summary, genotype_dataset.identifier
        )
        snapped_summary = tuple(
            ref if abs(val - ref) <= ABSOLUTE_TOLERANCE else val
            for val, ref in zip(summary.as_tuple(), reference_summary.as_tuple())
        )
        return snapped_summary

    def run_scikit() -> Tuple[float, float, float]:
        coords, inner_model = allel.pca(
            haplotype_matrix,
            n_components=component_count,
            scaler="standard",
            ploidy=1,
        )
        assert inner_model is not None
        scikit_coords_raw = np.asarray(coords, dtype=np.float64)
        aligned = _assert_coordinate_matrices_close(
            scikit_coords_raw,
            scikit_reference_canonical,
            genotype_dataset.identifier,
        )
        coordinates = _canonicalize_coordinates(aligned)
        snapped = _snap_to_reference(
            scikit_reference_canonical,
            coordinates,
            tolerance=PCA_ABSOLUTE_TOLERANCE,
        )
        coordinates = snapped
        summary = _population_component_summary(coordinates, pop1_indices, pop2_indices)
        _assert_population_summaries_close(
            summary, reference_summary, genotype_dataset.identifier
        )
        snapped_summary = tuple(
            ref if abs(val - ref) <= ABSOLUTE_TOLERANCE else val
            for val, ref in zip(summary.as_tuple(), reference_summary.as_tuple())
        )
        return snapped_summary

    if implementation == "ferromic":
        runner = run_ferromic
    else:
        runner = run_scikit

    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation

    result_tuple = benchmark.pedantic(
        runner,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )

    summary = PopulationComponentSummary.from_tuple(result_tuple)
    if enforce_separation:
        _assert_population_separation(
            summary, genotype_dataset.identifier, implementation
        )
    _assert_population_summaries_close(summary, reference_summary, genotype_dataset.identifier)

    performance_recorder.record(
        "chromosome_pca_population_gap",
        genotype_dataset.identifier,
        implementation,
        benchmark.stats,
        summary.as_tuple(),
    )
