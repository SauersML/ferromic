"""Regression test for audit #7: a haplotype with unknown codon degeneracy (N/gap at
positions 1-2) must NOT contribute its third base to fourfold-site pi."""
import numpy as np
from four_fold_pi import fourfold_columns, fourfold_locus_pi, locus_pi

def test_unknown_prefix_excluded():
    seqs = ["GCA", "NNG"]               # 'NNG' prefix unknown -> ineligible
    cols = list(fourfold_columns(seqs, 3))
    assert cols == [0]                  # GC* (Ala) is fourfold; column is considered
    pi, n = fourfold_locus_pi(seqs, cols)
    # Only 'GCA' is eligible -> a single called base -> pi undefined, NOT 1.0.
    assert n == 0 and np.isnan(pi), f"unknown-prefix haplotype leaked into pi: pi={pi}, n={n}"
    # The buggy path (count all haplotypes' 3rd base) would have given pi=1.0:
    buggy, _ = locus_pi(seqs, [c + 2 for c in cols])
    assert buggy == 1.0  # documents the old wrong behaviour

def test_eligible_pair_counts():
    seqs = ["GCA", "GCG"]               # both GC* fourfold -> both eligible
    cols = list(fourfold_columns(seqs, 3))
    pi, n = fourfold_locus_pi(seqs, cols)
    assert n == 1 and abs(pi - 1.0) < 1e-12, (pi, n)

if __name__ == "__main__":
    test_unknown_prefix_excluded()
    test_eligible_pair_counts()
    print("OK: fourfold pi excludes unknown-prefix haplotypes; eligible pairs counted")
