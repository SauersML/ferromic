import math

import pytest

import ferromic as fm


def build_variant(position, genotypes):
    """Helper to create a variant mapping for the Python API."""
    return {"position": position, "genotypes": genotypes}


def test_segregating_sites_counts_polymorphic_sites():
    variants = [
        build_variant(100, [[0, 0], [0, 1]]),
        build_variant(150, [[0, 0], [0, 0]]),
        build_variant(200, [[0, 1], [1, 1]]),
    ]

    assert fm.segregating_sites(variants) == 2


def test_watterson_theta_matches_rust_implementation():
    theta = fm.watterson_theta(3, 4, 100)
    expected = 3 / (1 + 1 / 2 + 1 / 3) / 100

    assert math.isclose(theta, expected, rel_tol=1e-12)


def test_watterson_theta_requires_multiple_samples():
    with pytest.raises(ValueError) as excinfo:
        fm.watterson_theta(1, 1, 100)

    assert "sample_count" in str(excinfo.value)


def test_adjusted_sequence_length_respects_allow_and_mask_regions():
    adjusted = fm.adjusted_sequence_length(
        1,
        100,
        allow=[(11, 20), (40, 60)],
        mask=[(45, 50)],
    )

    assert adjusted == 25


def test_population_rejects_non_positive_sequence_length():
    with pytest.raises(ValueError) as excinfo:
        fm.Population("demo", [], [], 0)

    assert "sequence_length" in str(excinfo.value)


def test_inversion_allele_frequency_counts_haplotypes():
    sample_map = {
        "sampleA": (0, 1),
        "sampleB": (1, 1),
        "sampleC": (2, 255),  # ignored because they are not 0/1 alleles
    }

    frequency = fm.inversion_allele_frequency(sample_map)

    assert frequency == pytest.approx(0.75)
