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

SEQUENCE_LENGTH = 3


def make_variants():
    """Return a small diploid genotype panel shared with scikit-allel."""
    return [
        {
            "position": 0,
            "genotypes": [
                [0, 0],
                [0, 0],
                [1, 1],
                [1, 1],
            ],
        },
        {
            "position": 1,
            "genotypes": [
                [0, 1],
                [0, 0],
                [0, 1],
                [0, 1],
            ],
        },
        {
            "position": 2,
            "genotypes": [
                [0, 0],
                [0, 1],
                [0, 1],
                [1, 1],
            ],
        },
    ]


def build_population(pop_id, sample_indices, variants):
    haplotypes = [
        (sample_index, haplotype_side)
        for sample_index in sample_indices
        for haplotype_side in (0, 1)
    ]
    return {
        "id": pop_id,
        "haplotypes": haplotypes,
        "variants": variants,
        "sequence_length": SEQUENCE_LENGTH,
        "sample_names": SAMPLE_NAMES,
    }


@pytest.fixture()
def hudson_dataset():
    variants = make_variants()
    genotype_array = allel.GenotypeArray([variant["genotypes"] for variant in variants])
    subpops = [POP1_SAMPLES, POP2_SAMPLES]
    allele_counts_1 = genotype_array.count_alleles(subpop=subpops[0])
    allele_counts_2 = genotype_array.count_alleles(subpop=subpops[1])
    numerator, denominator = allel.hudson_fst(allele_counts_1, allele_counts_2)

    return {
        "variants": variants,
        "population1": build_population("pop1", POP1_SAMPLES, variants),
        "population2": build_population("pop2", POP2_SAMPLES, variants),
        "numerator": numerator,
        "denominator": denominator,
    }


def test_hudson_fst_matches_scikit_allel_ratio_of_sums(hudson_dataset):
    result = fm.hudson_fst(hudson_dataset["population1"], hudson_dataset["population2"])

    numerator = hudson_dataset["numerator"]
    denominator = hudson_dataset["denominator"]
    expected_fst = float(numerator.sum() / denominator.sum())
    expected_dxy = float(denominator.sum() / SEQUENCE_LENGTH)

    print(
        "hudson_fst aggregate:",
        {
            "fst": {"actual": result.fst, "expected": expected_fst},
            "d_xy": {"actual": result.d_xy, "expected": expected_dxy},
        },
    )

    assert result.fst == pytest.approx(expected_fst, rel=1e-12)
    assert result.d_xy == pytest.approx(expected_dxy, rel=1e-12)


def test_hudson_fst_site_components_align_with_scikit_allel(hudson_dataset):
    region = (0, len(hudson_dataset["variants"]) - 1)
    result, sites = fm.hudson_fst_with_sites(
        hudson_dataset["population1"],
        hudson_dataset["population2"],
        region,
    )

    numerator = hudson_dataset["numerator"]
    denominator = hudson_dataset["denominator"]
    expected_fst = float(numerator.sum() / denominator.sum())
    print(
        "hudson_fst_with_sites aggregate:",
        {"fst": {"actual": result.fst, "expected": expected_fst}},
    )

    assert result.fst == pytest.approx(expected_fst, rel=1e-12)

    informative_sites = [
        site
        for site in sites
        if site.numerator_component is not None and site.denominator_component is not None
    ]
    assert len(informative_sites) == len(hudson_dataset["variants"])

    for idx, site in enumerate(informative_sites):
        position = hudson_dataset["variants"][idx]["position"] + 1
        expected_num = float(numerator[idx])
        expected_den = float(denominator[idx])
        expected_site_fst = expected_num / expected_den

        assert site.position == position
        print(
            "hudson_fst_with_sites site:",
            {
                "position": position,
                "numerator": {"actual": site.numerator_component, "expected": expected_num},
                "denominator": {
                    "actual": site.denominator_component,
                    "expected": expected_den,
                },
                "fst": {"actual": site.fst, "expected": expected_site_fst},
            },
        )
        assert site.numerator_component == pytest.approx(expected_num, rel=1e-12)
        assert site.denominator_component == pytest.approx(expected_den, rel=1e-12)
        assert site.fst == pytest.approx(expected_site_fst, rel=1e-12)
