"""Integration checks for diversity statistics against scikit-allel."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import ferromic as fm

allel = pytest.importorskip("allel")

SAMPLE_NAMES = [
    "pop1_individual_1",
    "pop1_individual_2",
    "pop2_individual_1",
    "pop2_individual_2",
]

POP1_SAMPLES = [0, 1]
POP2_SAMPLES = [2, 3]

SEQUENCE_LENGTH = 10


def build_variants():
    """Return a diploid variant panel shared between tests and libraries."""

    return [
        {
            "position": 0,
            "genotypes": [
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
        },
        {
            "position": 3,
            "genotypes": [
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 0],
            ],
        },
        {
            "position": 5,
            "genotypes": [
                [0, 0],
                [0, 1],
                [0, 1],
                [1, 1],
            ],
        },
        {
            "position": 7,
            "genotypes": [
                [0, 1],
                [1, 1],
                None,
                [0, 1],
            ],
        },
    ]


def build_haplotypes(sample_indices):
    return [
        (sample_index, haplotype_side)
        for sample_index in sample_indices
        for haplotype_side in (0, 1)
    ]


def build_population(pop_id, sample_indices, variants):
    return {
        "id": pop_id,
        "haplotypes": build_haplotypes(sample_indices),
        "variants": copy.deepcopy(variants),
        "sequence_length": SEQUENCE_LENGTH,
        "sample_names": SAMPLE_NAMES,
    }


def variants_to_genotype_array(variants):
    calls = []
    for variant in variants:
        site_calls = []
        for genotype in variant["genotypes"]:
            if genotype is None:
                site_calls.append([-1, -1])
            else:
                site_calls.append(list(genotype))
        calls.append(site_calls)
    return allel.GenotypeArray(np.array(calls, dtype=np.int16))


@pytest.fixture()
def diversity_dataset():
    variants = build_variants()
    genotype_array = variants_to_genotype_array(variants)
    allele_counts_pop1 = genotype_array.count_alleles(subpop=POP1_SAMPLES)
    allele_counts_pop2 = genotype_array.count_alleles(subpop=POP2_SAMPLES)
    allele_counts_all = genotype_array.count_alleles()

    return {
        "variants": variants,
        "sequence_length": SEQUENCE_LENGTH,
        "haplotypes": {
            "pop1": build_haplotypes(POP1_SAMPLES),
            "pop2": build_haplotypes(POP2_SAMPLES),
            "combined": build_haplotypes(POP1_SAMPLES + POP2_SAMPLES),
        },
        "allele_counts": {
            "pop1": allele_counts_pop1,
            "pop2": allele_counts_pop2,
            "combined": allele_counts_all,
        },
        "positions": [variant["position"] + 1 for variant in variants],
        "populations": {
            "pop1": build_population("pop1", POP1_SAMPLES, variants),
            "pop2": build_population("pop2", POP2_SAMPLES, variants),
        },
    }


def test_nucleotide_diversity_matches_scikit_allel(diversity_dataset):
    dataset = diversity_dataset
    sequence_length = dataset["sequence_length"]

    expected_pop1 = float(
        np.nansum(allel.mean_pairwise_difference(dataset["allele_counts"]["pop1"]))
        / sequence_length
    )
    expected_pop2 = float(
        np.nansum(allel.mean_pairwise_difference(dataset["allele_counts"]["pop2"]))
        / sequence_length
    )
    expected_combined = float(
        np.nansum(allel.mean_pairwise_difference(dataset["allele_counts"]["combined"]))
        / sequence_length
    )

    actual_pop1 = fm.nucleotide_diversity(
        dataset["variants"], dataset["haplotypes"]["pop1"], sequence_length
    )
    actual_pop2 = fm.nucleotide_diversity(
        dataset["variants"], dataset["haplotypes"]["pop2"], sequence_length
    )
    actual_combined = fm.nucleotide_diversity(
        dataset["variants"], dataset["haplotypes"]["combined"], sequence_length
    )

    print(
        "nucleotide_diversity results:",
        {
            "pop1": {"actual": actual_pop1, "expected": expected_pop1},
            "pop2": {"actual": actual_pop2, "expected": expected_pop2},
            "combined": {"actual": actual_combined, "expected": expected_combined},
        },
    )

    assert actual_pop1 == pytest.approx(expected_pop1, rel=1e-12)
    assert actual_pop2 == pytest.approx(expected_pop2, rel=1e-12)
    assert actual_combined == pytest.approx(expected_combined, rel=1e-12)


def test_per_site_diversity_pi_aligns_with_scikit_allel(diversity_dataset):
    dataset = diversity_dataset
    region = (0, dataset["sequence_length"] - 1)
    sites = fm.per_site_diversity(
        dataset["variants"], dataset["haplotypes"]["pop1"], region
    )
    site_by_position = {site.position: site for site in sites}

    per_variant_pi = np.nan_to_num(
        allel.mean_pairwise_difference(dataset["allele_counts"]["pop1"]), nan=0.0
    )

    for position, expected_pi in zip(dataset["positions"], per_variant_pi):
        assert position in site_by_position
        actual_pi = site_by_position[position].pi
        print(
            f"per_site_diversity position={position} actual={actual_pi} expected={expected_pi}"
        )
        assert actual_pi == pytest.approx(expected_pi, rel=1e-12)


def test_hudson_dxy_matches_scikit_allel(diversity_dataset):
    dataset = diversity_dataset
    result = fm.hudson_dxy(
        dataset["populations"]["pop1"], dataset["populations"]["pop2"]
    )

    expected_dxy = float(
        np.nansum(
            allel.mean_pairwise_difference_between(
                dataset["allele_counts"]["pop1"], dataset["allele_counts"]["pop2"]
            )
        )
        / dataset["sequence_length"]
    )

    print(
        "hudson_dxy result:",
        {"actual": result.d_xy, "expected": expected_dxy},
    )

    assert result.d_xy == pytest.approx(expected_dxy, rel=1e-12)
