"""Benchmarks comparing ferromic's Python API against scikit-allel."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pytest

import ferromic as fm

allel = pytest.importorskip("allel")

from pytest_benchmark.fixture import BenchmarkFixture
from pytest_benchmark.stats import Metadata


@dataclass(frozen=True)
class DatasetConfig:
    """Parameters controlling the synthetic cohorts used in benchmarks."""

    label: str
    variant_count: int
    sample_count: int
    divergence_scale: float
    benchmark_rounds: int = 5

    @property
    def identifier(self) -> str:
        return f"{self.label}_variants_{self.variant_count}_samples_{self.sample_count}"


DATASET_CONFIGS: Sequence[DatasetConfig] = (
    DatasetConfig("pilot_panel", 512, 48, 0.02),
    DatasetConfig("regional_panel", 4096, 96, 0.05),
    DatasetConfig("chromosome_arm", 16384, 128, 0.08, benchmark_rounds=4),
    DatasetConfig("deep_cohort", 65536, 256, 0.1, benchmark_rounds=3),
)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Reusable data structures shared across benchmarks and equivalence tests."""

    identifier: str
    variants: List[Dict[str, object]]
    haplotypes: List[Tuple[int, int]]
    sample_names: List[str]
    sequence_length: int
    positions: np.ndarray
    genotype_array: "allel.GenotypeArray"
    allele_counts_total: "allel.AlleleCountsArray"
    allele_counts_pop1: "allel.AlleleCountsArray"
    allele_counts_pop2: "allel.AlleleCountsArray"
    pop1: Dict[str, object]
    pop2: Dict[str, object]
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
    config: DatasetConfig = request.param
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
    sequence_stop = int(positions[-1]) + 1 if config.variant_count else 0
    sequence_length = sequence_stop - sequence_start

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
            stop=sequence_stop,
        )
    )
    expected_pi_pop1 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop1,
            start=sequence_start,
            stop=sequence_stop,
        )
    )
    expected_pi_pop2 = float(
        allel.sequence_diversity(
            positions,
            allele_counts_pop2,
            start=sequence_start,
            stop=sequence_stop,
        )
    )
    expected_theta = float(
        allel.watterson_theta(
            positions,
            allele_counts_total,
            start=sequence_start,
            stop=sequence_stop,
        )
    )

    pop1 = _build_population("population_1", pop1_indices, haplotypes, variants, sequence_length, sample_names)
    pop2 = _build_population("population_2", pop2_indices, haplotypes, variants, sequence_length, sample_names)

    return BenchmarkDataset(
        identifier=config.identifier,
        variants=variants,
        haplotypes=haplotypes,
        sample_names=sample_names,
        sequence_length=sequence_length,
        positions=positions,
        genotype_array=genotype_array,
        allele_counts_total=allele_counts_total,
        allele_counts_pop1=allele_counts_pop1,
        allele_counts_pop2=allele_counts_pop2,
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
    population_id: str,
    sample_indices: Iterable[int],
    haplotypes: Sequence[Tuple[int, int]],
    variants: Sequence[Dict[str, object]],
    sequence_length: int,
    sample_names: Sequence[str],
) -> Dict[str, object]:
    haplotype_lookup = {
        (sample_index, haplotype_side)
        for sample_index in sample_indices
        for haplotype_side in (0, 1)
    }
    filtered_haplotypes = [h for h in haplotypes if h in haplotype_lookup]
    return {
        "id": population_id,
        "haplotypes": filtered_haplotypes,
        "variants": variants,
        "sequence_length": sequence_length,
        "sample_names": sample_names,
    }


@pytest.fixture(scope="module")
def performance_recorder():
    store: Dict[str, Dict[str, Dict[str, "Metadata"]]] = defaultdict(lambda: defaultdict(dict))
    yield store
    for metric, dataset_map in store.items():
        for dataset_id, stats_by_library in dataset_map.items():
            missing = {"ferromic", "scikit-allel"} - set(stats_by_library)
            assert not missing, f"Missing benchmarks for {metric} on {dataset_id}: {sorted(missing)}"
            ferromic_stats = stats_by_library["ferromic"].stats
            competitor_stats = stats_by_library["scikit-allel"].stats
            if competitor_stats.mean == 0:
                continue
            ratio = ferromic_stats.mean / competitor_stats.mean
            assert math.isfinite(ratio), "Performance ratio must be finite"
            stats_by_library["ferromic"].extra_info[f"relative_to_scikit_{dataset_id}_{metric}"] = ratio


# ---------------------------------------------------------------------------
# Equivalence checks.
# ---------------------------------------------------------------------------


def test_segregating_sites_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.segregating_sites(genotype_dataset.variants)
    scikit_value = int(genotype_dataset.allele_counts_total.is_segregating().sum())

    assert ferromic_value == genotype_dataset.expected_segregating_sites
    assert ferromic_value == scikit_value


def test_nucleotide_diversity_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.nucleotide_diversity(
        genotype_dataset.variants,
        genotype_dataset.haplotypes,
        genotype_dataset.sequence_length,
    )

    assert ferromic_value == pytest.approx(
        genotype_dataset.expected_nucleotide_diversity,
        rel=5e-4,
    )


def test_watterson_theta_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    ferromic_value = fm.watterson_theta(
        genotype_dataset.expected_segregating_sites,
        genotype_dataset.haplotype_count,
        genotype_dataset.sequence_length,
    )

    assert ferromic_value == pytest.approx(
        genotype_dataset.expected_watterson_theta,
        rel=5e-4,
    )


def test_hudson_fst_matches_scikit_allel(genotype_dataset: BenchmarkDataset):
    result = fm.hudson_fst(genotype_dataset.pop1, genotype_dataset.pop2)

    assert result.fst == pytest.approx(genotype_dataset.expected_hudson_fst, rel=1e-9, abs=1e-12)
    assert result.d_xy == pytest.approx(genotype_dataset.expected_hudson_dxy, rel=1e-9, abs=1e-12)


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
            return fm.segregating_sites(genotype_dataset.variants)
    else:
        allele_counts = genotype_dataset.allele_counts_total

        def run() -> int:
            return int(allele_counts.is_segregating().sum())

    result = benchmark.pedantic(
        run,
        iterations=1,
        rounds=genotype_dataset.benchmark_rounds,
    )
    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation
    assert result == genotype_dataset.expected_segregating_sites
    performance_recorder["segregating_sites"][genotype_dataset.identifier][implementation] = benchmark.stats


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_nucleotide_diversity(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        def run() -> float:
            return fm.nucleotide_diversity(
                genotype_dataset.variants,
                genotype_dataset.haplotypes,
                genotype_dataset.sequence_length,
            )
    else:
        allele_counts = genotype_dataset.allele_counts_total
        positions = genotype_dataset.positions
        start = int(positions[0]) if len(positions) else 0
        stop = start + genotype_dataset.sequence_length

        def run() -> float:
            return float(
                allel.sequence_diversity(
                    positions,
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
    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation
    assert result == pytest.approx(
        genotype_dataset.expected_nucleotide_diversity,
        rel=5e-4,
    )
    performance_recorder["nucleotide_diversity"][genotype_dataset.identifier][implementation] = benchmark.stats


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
            genotype_dataset.pop1["haplotypes"],
            genotype_dataset.allele_counts_pop1,
            genotype_dataset.expected_nucleotide_diversity_pop1,
            genotype_dataset.pop1["id"],
        ),
        "pop2": (
            genotype_dataset.pop2["haplotypes"],
            genotype_dataset.allele_counts_pop2,
            genotype_dataset.expected_nucleotide_diversity_pop2,
            genotype_dataset.pop2["id"],
        ),
    }
    haplotypes, allele_counts, expected_value, population_id = populations[population_key]

    if implementation == "ferromic":
        def run() -> float:
            return fm.nucleotide_diversity(
                genotype_dataset.variants,
                haplotypes,
                genotype_dataset.sequence_length,
            )
    else:
        positions = genotype_dataset.positions
        start = int(positions[0]) if len(positions) else 0
        stop = start + genotype_dataset.sequence_length

        def run() -> float:
            return float(
                allel.sequence_diversity(
                    positions,
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
    assert result == pytest.approx(expected_value, rel=5e-4)
    performance_recorder["nucleotide_diversity_population"][dataset_key][implementation] = benchmark.stats


@pytest.mark.parametrize("implementation", ["ferromic", "scikit-allel"])
def test_benchmark_watterson_theta(
    benchmark: "BenchmarkFixture",
    genotype_dataset: BenchmarkDataset,
    performance_recorder,
    implementation: str,
) -> None:
    if implementation == "ferromic":
        def run() -> float:
            return fm.watterson_theta(
                genotype_dataset.expected_segregating_sites,
                genotype_dataset.haplotype_count,
                genotype_dataset.sequence_length,
            )
    else:
        allele_counts = genotype_dataset.allele_counts_total
        positions = genotype_dataset.positions
        start = int(positions[0]) if len(positions) else 0
        stop = start + genotype_dataset.sequence_length

        def run() -> float:
            return float(
                allel.watterson_theta(
                    positions,
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
    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation
    assert result == pytest.approx(
        genotype_dataset.expected_watterson_theta,
        rel=5e-4,
    )
    performance_recorder["watterson_theta"][genotype_dataset.identifier][implementation] = benchmark.stats


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
    benchmark.extra_info["dataset"] = genotype_dataset.identifier
    benchmark.extra_info["implementation"] = implementation
    benchmark.extra_info["fst"] = fst_value
    benchmark.extra_info["d_xy"] = d_xy_value
    assert fst_value == pytest.approx(
        genotype_dataset.expected_hudson_fst,
        rel=1e-9,
        abs=1e-12,
    )
    assert d_xy_value == pytest.approx(
        genotype_dataset.expected_hudson_dxy,
        rel=1e-9,
        abs=1e-12,
    )
    performance_recorder["hudson_fst"][genotype_dataset.identifier][implementation] = benchmark.stats
