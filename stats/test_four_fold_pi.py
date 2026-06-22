"""Regression tests for stats/four_fold_pi.py.

Focus: the per-haplotype eligibility mask at fourfold third-codon positions
(audit BUG #7). A haplotype whose positions 1-2 are gap/N has unknown codon
degeneracy, so its third base cannot be established as a synonymous (fourfold)
allele and must be excluded from the fourfold-site pi calculation.
"""

import numpy as np

from stats.four_fold_pi import fourfold_columns, locus_pi, site_pi


def test_ineligible_haplotype_excluded_from_fourfold_pi():
    """['GCA', 'NNG']: the 2nd haplotype's first two bases (N, N) do not
    establish a fourfold codon, so its third base ('G') must NOT enter the
    fourfold pi at column 2. Only 'GCA' (Ala prefix 'GC') is eligible, leaving a
    single called base -> uncallable site -> pi excluded entirely.

    Before the fix the column was marked fourfold (the N/N haplotype was ignored
    in the decision) yet 'A' and 'G' were both folded into pi, giving pi=1.0."""
    seqs = ["GCA", "NNG"]
    cols = list(fourfold_columns(seqs, 3))

    # The codon is still a valid fourfold *site* (no called non-fourfold prefix),
    # so the third position (index 2) is yielded, but with an eligibility mask
    # that excludes the N/N haplotype.
    assert len(cols) == 1
    col_idx, eligible = cols[0]
    assert col_idx == 2
    assert eligible == [True, False]

    # With only one eligible called base, the site is uncallable -> contributes
    # nothing to pi (pi over zero callable sites is NaN, n=0).
    pi, n = locus_pi(seqs, cols)
    assert n == 0
    assert np.isnan(pi)


def test_two_eligible_haplotypes_give_expected_pi():
    """Both haplotypes establish a fourfold (Ala 'GC') codon; their third bases
    differ (A vs G), so the fourfold site is callable with pi = n/(n-1)*(1-sum
    p_i^2) = 2/1*(1-0.5) = 1.0."""
    seqs = ["GCA", "GCG"]
    cols = list(fourfold_columns(seqs, 3))
    assert len(cols) == 1
    col_idx, eligible = cols[0]
    assert eligible == [True, True]
    pi, n = locus_pi(seqs, cols)
    assert n == 1
    assert pi == site_pi(["A", "G"]) == 1.0


def test_called_nonfourfold_prefix_drops_codon():
    """If any *called* haplotype carries a non-fourfold prefix the third base is
    not a fourfold site for the sample, so the codon is dropped entirely.
    'ATG' (Met) has prefix 'AT' which is not fourfold-degenerate."""
    seqs = ["GCA", "ATG"]
    cols = list(fourfold_columns(seqs, 3))
    assert cols == []


def test_eligibility_mask_excludes_ineligible_but_keeps_pi_of_eligible():
    """Three haplotypes: two establish a fourfold Ala codon (eligible) with
    differing third bases, one has N/N at positions 1-2 (ineligible) and a third
    base that would otherwise inflate diversity. pi must be computed from the two
    eligible haplotypes only."""
    seqs = ["GCA", "GCG", "NNA"]
    cols = list(fourfold_columns(seqs, 3))
    assert len(cols) == 1
    _, eligible = cols[0]
    assert eligible == [True, True, False]

    pi, n = locus_pi(seqs, cols)
    assert n == 1
    # Only A (from GCA) and G (from GCG) count -> pi = 1.0, regardless of the
    # ineligible 'A' third base of the NNA haplotype.
    assert pi == 1.0


def test_locus_pi_accepts_bare_indices_for_whole_cds():
    """whole_cds_pi passes bare column indices (range), exercising the non-tuple
    branch of locus_pi: every haplotype's base is used."""
    seqs = ["AAAA", "AAAT"]
    pi, n = locus_pi(seqs, range(4))
    # Columns 0-2 are monomorphic (pi=0); column 3 is A/T -> pi=1.0.
    assert n == 4
    assert pi == np.mean([0.0, 0.0, 0.0, 1.0])
