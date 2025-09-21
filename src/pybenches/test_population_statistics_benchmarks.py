"""Benchmarks comparing ferromic's Python API against scikit-allel."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pytest

import ferromic as fm

try:  # scikit-allel is required for fair benchmark comparisons.
    import allel
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ModuleNotFoundError(
        "scikit-allel must be installed for population statistic benchmarks. "
        "Install ferromic with the 'test' extra (pip install .[test]) so both "
        "implementations are exercised."
    ) from exc

from pytest_benchmark.fixture import BenchmarkFixture
from pytest_benchmark.stats import Metadata


ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """Parameters controlling the synthetic cohorts used in benchmarks."""

    label: str
    variant_count: int
    sample_count: int
    divergence_scale: float
    benchmark_rounds: int = 5

    @property
    def identifier(self) -> str:
        return f"{self.label}_variants_{self.variant_count}_samples_{self.sample_count}"

    def build(self) -> "BenchmarkDataset":
        return _build_synthetic_dataset(self)


@dataclass(frozen=True)
class RealDatasetConfig:
    """Configuration describing a VCF-backed benchmark cohort."""

    label: str
    vcf_filename: str
    benchmark_rounds: int = 3
    pop1_size: int | None = None

    @property
    def identifier(self) -> str:
        return f"{self.label}_vcf"

    def build(self) -> "BenchmarkDataset":
        return _build_real_dataset(self)


DATASET_CONFIGS: Sequence[Union[SyntheticDatasetConfig, RealDatasetConfig]] = (
    SyntheticDatasetConfig("pilot_panel", 512, 48, 0.02),
    SyntheticDatasetConfig("regional_panel", 4096, 96, 0.05),
    SyntheticDatasetConfig("chromosome_arm", 16384, 128, 0.08, benchmark_rounds=4),
    SyntheticDatasetConfig("deep_cohort", 65536, 256, 0.1, benchmark_rounds=3),
    RealDatasetConfig("ag1000g_excerpt", "sample.vcf.gz", benchmark_rounds=2),
)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Reusable data structures shared across benchmarks and equivalence tests."""

    identifier: str
    variants: List[Dict[str, object]]
    haplotypes: List[Tuple[int, int]]
    sample_names: List[str]
    sequence_start: int
    sequence_stop_inclusive: int
    sequence_length: int
    positions: np.ndarray
    genotype_array: "allel.GenotypeArray"
    allele_counts_total: "allel.AlleleCountsArray"
    allele_counts_pop1: "allel.AlleleCountsArray"
    allele_counts_pop2: "allel.AlleleCountsArray"
    population: "fm.Population"
    pop1: "fm.Population"
    pop2: "fm.Population"
    expected_segregating_sites: int
    expected_nucleotide_diversity: float
    expected_nucleotide_diversity_pop1: float
    expected_nucleotide_diversity_pop2: float
    expected_watterson_theta: float
    expected_hudson_fst: float
    expected_hudson_dxy: float
    haplotype_count: int
    benchmark_rounds: int


@pytest.fixture(scope="module", params=DATASET_CONFIGS, ids=lambda config: config.identifier)
def genotype_dataset(request: pytest.FixtureRequest) -> BenchmarkDataset:
    config = request.param
    dataset: BenchmarkDataset = config.build()
    return dataset


def _build_synthetic_dataset(config: SyntheticDatasetConfig) -> BenchmarkDataset:
    sample_count = config.sample_count
    if sample_count % 2:
        raise ValueError("sample count must be even so we can split populations evenly")

    rng = np.random.default_rng(seed=config.variant_count + sample_count)
    half = sample_count // 2

    base_freq = rng.beta(0.8, 0.8, size=config.variant_count)
    divergence = rng.normal(0.0, config.divergence_scale, size=config.variant_count)
    pop1_freq = np.clip(base_freq + divergence, 0.001, 0.999)
    pop2_freq = np.clip(base_freq - divergence, 0.001, 0.999)

    pop1_haplotypes = rng.binomial(
        1,
        pop1_freq[:, None],
        size=(config.variant_count, half * 2),
    ).astype(np.int8)
    pop2_haplotypes = rng.binomial(
        1,
        pop2_freq[:, None],
        size=(config.variant_count, half * 2),
    ).astype(np.int8)

    genotypes = np.concatenate(
        [
            pop1_haplotypes.reshape(config.variant_count, half, 2),
            pop2_haplotypes.reshape(config.variant_count, half, 2),
        ],
        axis=1,
    )

    # Ensure at least two informative variants for stability across runs.
    if config.variant_count:
        genotypes[0, :half, :] = 0
        genotypes[0, half:, :] = 1
    if config.variant_count > 1:
        genotypes[1, :half, 0] = 0
        genotypes[1, :half, 1] = 1
        genotypes[1, half:, :] = 1

    if config.variant_count:
        increments = rng.integers(1, 50, size=config.variant_count, dtype=np.int64)
        positions = np.cumsum(increments, dtype=np.int64)
    else:
        positions = np.array([], dtype=np.int64)

    sequence_start = int(positions[0]) if config.variant_count else 0
    sequence_stop_exclusive = int(positions[-1]) + 1 if config.variant_count else sequence_start
    sequence_length = sequence_stop_exclusive - sequence_start
    sequence_stop_inclusive = (
        sequence_stop_exclusive - 1 if sequence_length > 0 else sequence_start
    )

    variants = [
        {"position": int(position), "genotypes": genotypes[idx].tolist()}
        for idx, position in enumerate(positions)
    ]

    haplotypes = [
        (sample_index, haplotype_side)
        for sample_index in range(sample_count)
        for haplotype_side in (0, 1)
    ]
    sample_names = [f"sample_{idx}" for idx in range(sample_count)]

    population = fm.Population.from_numpy(
        "all_samples",
        genotypes,
        positions,
        haplotypes,
        sequence_length,
        sample_names=sample_names,
    )

    genotype_array = allel.GenotypeArray(genotypes)
    allele_counts_total = genotype_array.count_alleles(max_allele=2)

    pop1_indices = list(range(half))
    pop2_indices = list(range(half, sample_count))
    allele_counts_pop1 = genotype_array.count_alleles(subpop=pop1_indices, max_allele=2)
    allele_counts_pop2 = genotype_array.count_alleles(subpop=pop2_indices, max_allele=2)

    numerator, denominator = allel.hudson_fst(allele_counts_pop1, allele_counts_pop2)
    fst = float(numerator.sum() / denominator.sum()) if float(denominator.sum()) else math.nan
    d_xy = float(denominator.sum() / sequence_length) if sequence_length else math.nan

    expected_pi_total = float(
        allel.sequence_diversity(
            positions,
            allele_counts_total,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_pi_pop1 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop1,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_pi_pop2 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop2,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_theta = float(
        allel.watterson_theta(
            positions,
            allele_counts_total,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )

    pop1 = _build_population(population, "population_1", pop1_indices, haplotypes)
    pop2 = _build_population(population, "population_2", pop2_indices, haplotypes)

    return BenchmarkDataset(
        identifier=config.identifier,
        variants=variants,
        haplotypes=haplotypes,
        sample_names=sample_names,
        sequence_start=sequence_start,
        sequence_stop_inclusive=sequence_stop_inclusive,
        sequence_length=sequence_length,
        positions=positions,
        genotype_array=genotype_array,
        allele_counts_total=allele_counts_total,
        allele_counts_pop1=allele_counts_pop1,
        allele_counts_pop2=allele_counts_pop2,
        population=population,
        pop1=pop1,
        pop2=pop2,
        expected_segregating_sites=int(allele_counts_total.is_segregating().sum()),
        expected_nucleotide_diversity=expected_pi_total,
        expected_nucleotide_diversity_pop1=expected_pi_pop1,
        expected_nucleotide_diversity_pop2=expected_pi_pop2,
        expected_watterson_theta=expected_theta,
        expected_hudson_fst=fst,
        expected_hudson_dxy=d_xy,
        haplotype_count=sample_count * 2,
        benchmark_rounds=config.benchmark_rounds,
    )


def _build_real_dataset(config: RealDatasetConfig) -> BenchmarkDataset:
    data_path = Path(allel.__file__).resolve().parent / "test" / "data" / config.vcf_filename
    callset = allel.read_vcf(str(data_path), alt_number=1)

    genotype_array = allel.GenotypeArray(callset["calldata/GT"])
    positions = np.asarray(callset["variants/POS"], dtype=np.int64)
    if genotype_array.size == 0 or positions.size == 0:
        raise ValueError(f"VCF file {data_path} does not contain genotype information")

    completeness_mask = ~genotype_array.is_missing().any(axis=1)
    genotype_array = genotype_array.compress(completeness_mask, axis=0)
    positions = positions[completeness_mask]
    if genotype_array.shape[0] == 0:
        raise ValueError(f"VCF file {data_path} does not contain fully called variants")

    sequence_start = int(positions[0])
    sequence_stop_exclusive = int(positions[-1]) + 1
    sequence_length = sequence_stop_exclusive - sequence_start
    sequence_stop_inclusive = sequence_stop_exclusive - 1 if sequence_length > 0 else sequence_start

    sample_names = list(map(str, callset["samples"]))
    sample_count = len(sample_names)
    if sample_count < 2:
        raise ValueError("at least two samples are required to define populations")

    pop1_size = config.pop1_size if config.pop1_size is not None else sample_count // 2
    pop1_size = max(1, min(pop1_size, sample_count - 1))
    pop1_indices = list(range(pop1_size))
    pop2_indices = list(range(pop1_size, sample_count))

    haplotypes = [
        (sample_index, haplotype_side)
        for sample_index in range(sample_count)
        for haplotype_side in (0, 1)
    ]

    variants = [
        {"position": int(position), "genotypes": genotype_array[idx].tolist()}
        for idx, position in enumerate(positions)
    ]

    genotypes_numpy = genotype_array.astype(np.int8)
    population = fm.Population.from_numpy(
        config.label,
        genotypes_numpy,
        positions,
        haplotypes,
        sequence_length,
        sample_names=sample_names,
    )

    allele_counts_total = genotype_array.count_alleles(max_allele=2)
    allele_counts_pop1 = genotype_array.count_alleles(subpop=pop1_indices, max_allele=2)
    allele_counts_pop2 = genotype_array.count_alleles(subpop=pop2_indices, max_allele=2)

    numerator, denominator = allel.hudson_fst(allele_counts_pop1, allele_counts_pop2)
    fst = float(numerator.sum() / denominator.sum()) if float(denominator.sum()) else math.nan
    d_xy = float(denominator.sum() / sequence_length) if sequence_length else math.nan

    expected_pi_total = float(
        allel.sequence_diversity(
            positions,
            allele_counts_total,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_pi_pop1 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop1,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_pi_pop2 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop2,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )
    expected_theta = float(
        allel.watterson_theta(
            positions,
            allele_counts_total,
            start=sequence_start,
            stop=sequence_stop_inclusive,
        )
    )

    pop1 = _build_population(population, "population_1", pop1_indices, haplotypes)
    pop2 = _build_population(population, "population_2", pop2_indices, haplotypes)

    return BenchmarkDataset(
        identifier=config.identifier,
        variants=variants,
        haplotypes=haplotypes,
        sample_names=sample_names,
        sequence_start=sequence_start,
        sequence_stop_inclusive=sequence_stop_inclusive,
        sequence_length=sequence_length,
        positions=positions,
        genotype_array=genotype_array,
        allele_counts_total=allele_counts_total,
        allele_counts_pop1=allele_counts_pop1,
        allele_counts_pop2=allele_counts_pop2,
        population=population,
        pop1=pop1,
        pop2=pop2,
        expected_segregating_sites=int(allele_counts_total.is_segregating().sum()),
        expected_nucleotide_diversity=expected_pi_total,
        expected_nucleotide_diversity_pop1=expected_pi_pop1,
        expected_nucleotide_diversity_pop2=expected_pi_pop2,
        expected_watterson_theta=expected_theta,
        expected_hudson_fst=fst,
        expected_hudson_dxy=d_xy,
        haplotype_count=sample_count * 2,
        benchmark_rounds=config.benchmark_rounds,
    )


def _build_population(
    base_population: "fm.Population",
    population_id: str,
    sample_indices: Iterable[int],
    haplotypes: Sequence[Tuple[int, int]],
) -> "fm.Population":
    haplotype_lookup = {
        (sample_index, haplotype_side)
        for sample_index in sample_indices
        for haplotype_side in (0, 1)
    }
    filtered_haplotypes = [h for h in haplotypes if h in haplotype_lookup]
    return base_population.with_haplotypes(population_id, filtered_haplotypes)


def _assert_float_close(actual: float, expected: float, metric: str, dataset_id: str) -> None:
    if math.isnan(expected):
        assert math.isnan(actual), f"{metric} expected NaN for {dataset_id}"
        return
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=ABSOLUTE_TOLERANCE,
    ), f"{metric} mismatch for {dataset_id}: {actual} != {expected}"


def _assert_results_equal(
    ferromic_result: Any,
    scikit_result: Any,
    metric: str,
    dataset_id: str,
) -> None:
    if isinstance(ferromic_result, tuple) and isinstance(scikit_result, tuple):
        assert len(ferromic_result) == len(
            scikit_result
        ), f"Tuple length mismatch for {metric} on {dataset_id}"
        for index, (ferro_value, scikit_value) in enumerate(
            zip(ferromic_result, scikit_result)
        ):
            _assert_results_equal(
                ferro_value,
                scikit_value,
                f"{metric}[{index}]",
                dataset_id,
            )
        return

    if isinstance(scikit_result, (int, np.integer)):
        assert ferromic_result == scikit_result, (
            f"{metric} mismatch for {dataset_id}: {ferromic_result} != {scikit_result}"
        )
        return

    if isinstance(scikit_result, str):
        assert ferromic_result == scikit_result, (
            f"{metric} mismatch for {dataset_id}: {ferromic_result!r} != {scikit_result!r}"
        )
        return

    ferromic_value = float(ferromic_result)
    scikit_value = float(scikit_result)
    _assert_float_close(ferromic_value, scikit_value, metric, dataset_id)


@dataclass
class BenchmarkRecord:
    """Benchmark statistics paired with the exact computed result."""

    stats: "Metadata"
    result: Any


@dataclass
class PerformanceRecorder:
    timings: Dict[str, Dict[str, Dict[str, BenchmarkRecord]]]

    def record(
        self,
        metric: str,
        dataset_id: str,
        implementation: str,
        stats: "Metadata",
        result: Any,
    ) -> None:
        self.timings[metric][dataset_id][implementation] = BenchmarkRecord(stats=stats, result=result)


@pytest.fixture(scope="module")
def performance_recorder():
    recorder = PerformanceRecorder(
        timings=defaultdict(lambda: defaultdict(dict)),
    )
    yield recorder
    for metric, dataset_map in recorder.timings.items():
        for dataset_id, stats_by_library in dataset_map.items():
            missing = {"ferromic", "scikit-allel"} - set(stats_by_library)
            assert not missing, f"Missing benchmarks for {metric} on {dataset_id}: {sorted(missing)}"

            ferromic_record = stats_by_library["ferromic"]
            competitor_record = stats_by_library["scikit-allel"]
            _assert_results_equal(
                ferromic_record.result,
                competitor_record.result,
                metric,
                dataset_id,
            )

            ferromic_stats = ferromic_record.stats.stats
            competitor_stats = competitor_record.stats.stats
            if competitor_stats.mean == 0:
                continue
            ratio = ferromic_stats.mean / competitor_stats.mean
            assert math.isfinite(ratio), "Performance ratio must be finite"
            ferromic_record.stats.extra_info[f"relative_to_scikit_{dataset_id}_{metric}"] = ratio


# ---------------------------------------------------------------------------
# Equivalence checks.
# ---------------------------------------------------------------------------


def test_segregating_sites_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.segregating_sites(genotype_dataset.variants)
    scikit_value = int(genotype_dataset.allele_counts_total.is_segregating().sum())

    assert ferromic_value == genotype_dataset.expected_segregating_sites
    assert ferromic_value == scikit_value
    assert genotype_dataset.population.segregating_sites() == scikit_value


def test_nucleotide_diversity_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.nucleotide_diversity(
        genotype_dataset.variants,
        genotype_dataset.haplotypes,
        genotype_dataset.sequence_length,
    )
    population_value = genotype_dataset.population.nucleotide_diversity()
    start = genotype_dataset.sequence_start
    stop = genotype_dataset.sequence_stop_inclusive
    scikit_value = float(
        allel.sequence_diversity(
            genotype_dataset.positions,
            genotype_dataset.allele_counts_total,
            start=start,
            stop=stop,
        )
    )

    dataset_id = genotype_dataset.identifier
    _assert_float_close(ferromic_value, scikit_value, "nucleotide_diversity", dataset_id)
    _assert_float_close(population_value, scikit_value, "nucleotide_diversity", dataset_id)
    _assert_float_close(
        scikit_value,
        genotype_dataset.expected_nucleotide_diversity,
        "nucleotide_diversity",
        dataset_id,
    )


def test_watterson_theta_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.watterson_theta(
        genotype_dataset.expected_segregating_sites,
        genotype_dataset.haplotype_count,
        genotype_dataset.sequence_length,
    )
    start = genotype_dataset.sequence_start
    stop = genotype_dataset.sequence_stop_inclusive
    scikit_value = float(
        allel.watterson_theta(
            genotype_dataset.positions,
            genotype_dataset.allele_counts_total,
            start=start,
            stop=stop,
        )
    )

    dataset_id = genotype_dataset.identifier
    _assert_float_close(ferromic_value, scikit_value, "watterson_theta", dataset_id)
    _assert_float_close(
        scikit_value,
        genotype_dataset.expected_watterson_theta,
        "watterson_theta",
        dataset_id,
    )


def test_hudson_fst_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    result = fm.hudson_fst(genotype_dataset.pop1, genotype_dataset.pop2)
    numerator, denominator = allel.hudson_fst(
        genotype_dataset.allele_counts_pop1,
        genotype_dataset.allele_counts_pop2,
    )
    denominator_sum = float(denominator.sum())
    fst_expected = (
        float(numerator.sum() / denominator_sum)
        if denominator_sum
        else math.nan
    )
    d_xy_expected = (
        float(denominator_sum / genotype_dataset.sequence_length)
        if denominator_sum and genotype_dataset.sequence_length
        else math.nan
    )

    dataset_id = genotype_dataset.identifier
    _assert_float_close(result.fst, fst_expected, "hudson_fst", dataset_id)
    _assert_float_close(result.d_xy, d_xy_expected, "hudson_dxy", dataset_id)
    _assert_float_close(
        fst_expected,
        genotype_dataset.expected_hudson_fst,
        "hudson_fst",
        dataset_id,
    )
    _assert_float_close(
        d_xy_expected,
        genotype_dataset.expected_hudson_dxy,
        "hudson_dxy",
        dataset_id,
    )


# ---------------------------------------------------------------------------
# Benchmark comparisons.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_segregating_sites(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        def run() -> int:
            return genotype_dataset.population.segregating_sites()
    else:
        allele_counts = genotype_dataset.allele_counts_total

        def run() -> int:
            return int(allele_counts.is_segregating().sum())

    result = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    dataset_id = genotype_dataset.identifier
    benchmark.extra_info["dataset"] = dataset_id
    benchmark.extra_info["implementation"] = implementation
    expected_value = genotype_dataset.expected_segregating_sites
    assert result == expected_value
    performance_recorder.record(
        "segregating_sites",
        dataset_id,
        implementation,
        benchmark.stats,
        int(result),
    )


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_nucleotide_diversity(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        def run() -> float:
            return genotype_dataset.population.nucleotide_diversity()
    else:
        allele_counts = genotype_dataset.allele_counts_total
        start = genotype_dataset.sequence_start
        stop = genotype_dataset.sequence_stop_inclusive

        def run() -> float:
            return float(
                allel.sequence_diversity(
                    genotype_dataset.positions,
                    allele_counts,
                    start=start,
                    stop=stop,
                )
            )

    result = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    dataset_id = genotype_dataset.identifier
    benchmark.extra_info["dataset"] = dataset_id
    benchmark.extra_info["implementation"] = implementation
    _assert_float_close(
        float(result),
        genotype_dataset.expected_nucleotide_diversity,
        "nucleotide_diversity",
        dataset_id,
    )
    performance_recorder.record(
        "nucleotide_diversity",
        dataset_id,
        implementation,
        benchmark.stats,
        float(result),
    )


@pytest.mark.parametrize("population_key", ["pop1", "pop2"])
@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_population_nucleotide_diversity(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
    population_key: str,
) -> None:
    populations = {
        "pop1": (
            "population_1",
            genotype_dataset.pop1,
            genotype_dataset.allele_counts_pop1,
            genotype_dataset.expected_nucleotide_diversity_pop1,
        ),
        "pop2": (
            "population_2",
            genotype_dataset.pop2,
            genotype_dataset.allele_counts_pop2,
            genotype_dataset.expected_nucleotide_diversity_pop2,
        ),
    }
    population_id, population_obj, allele_counts, expected_value = populations[population_key]

    if implementation == "ferromic":
        def run() -> float:
            return population_obj.nucleotide_diversity()
    else:
        start = genotype_dataset.sequence_start
        stop = genotype_dataset.sequence_stop_inclusive

        def run() -> float:
            return float(
                allel.sequence_diversity(
                    genotype_dataset.positions,
                    allele_counts,
                    start=start,
                    stop=stop,
                )
            )

    result = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    dataset_key = f"{genotype_dataset.identifier}_{population_id}"
    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["population"] = population_id
    benchmark.extra_info["implementation"] = implementation
    _assert_float_close(float(result), expected_value, "nucleotide_diversity", dataset_key)
    performance_recorder.record(
        "nucleotide_diversity_population",
        dataset_key,
        implementation,
        benchmark.stats,
        float(result),
    )


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_watterson_theta(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        population = genotype_dataset.population
        haplotype_count = genotype_dataset.haplotype_count
        sequence_length = genotype_dataset.sequence_length

        def run() -> float:
            segregating_sites = population.segregating_sites()
            return fm.watterson_theta(
                segregating_sites,
                haplotype_count,
                sequence_length,
            )
    else:
        allele_counts = genotype_dataset.allele_counts_total
        start = genotype_dataset.sequence_start
        stop = genotype_dataset.sequence_stop_inclusive

        def run() -> float:
            return float(
                allel.watterson_theta(
                    genotype_dataset.positions,
                    allele_counts,
                    start=start,
                    stop=stop,
                )
            )

    result = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    dataset_id = genotype_dataset.identifier
    benchmark.extra_info["dataset"] = dataset_id
    benchmark.extra_info["implementation"] = implementation
    _assert_float_close(
        float(result),
        genotype_dataset.expected_watterson_theta,
        "watterson_theta",
        dataset_id,
    )
    performance_recorder.record(
        "watterson_theta",
        dataset_id,
        implementation,
        benchmark.stats,
        float(result),
    )


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_hudson_fst_result(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        def run() -> Tuple[float, float]:
            result = fm.hudson_fst(genotype_dataset.pop1, genotype_dataset.pop2)
            return result.fst, result.d_xy
    else:
        allele_counts_pop1 = genotype_dataset.allele_counts_pop1
        allele_counts_pop2 = genotype_dataset.allele_counts_pop2
        sequence_length = genotype_dataset.sequence_length

        def run() -> Tuple[float, float]:
            numerator, denominator = allel.hudson_fst(allele_counts_pop1, allele_counts_pop2)
            fst = float(numerator.sum() / denominator.sum())
            d_xy = float(denominator.sum() / sequence_length) if sequence_length else math.nan
            return fst, d_xy

    fst_value, d_xy_value = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    dataset_id = genotype_dataset.identifier
    benchmark.extra_info["dataset"] = dataset_id
    benchmark.extra_info["implementation"] = implementation
    benchmark.extra_info["fst"] = fst_value
    benchmark.extra_info["d_xy"] = d_xy_value
    _assert_float_close(float(fst_value), genotype_dataset.expected_hudson_fst, "hudson_fst", dataset_id)
    _assert_float_close(float(d_xy_value), genotype_dataset.expected_hudson_dxy, "hudson_dxy", dataset_id)
    performance_recorder.record(
        "hudson_fst",
        dataset_id,
        implementation,
        benchmark.stats,
        (float(fst_value), float(d_xy_value)),
    )
