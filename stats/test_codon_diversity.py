"""Unit tests for the shared codon-diversity source of truth (stdlib + numpy only).

Run: python -m pytest stats/test_codon_diversity.py   (or: python stats/test_codon_diversity.py)
These pin the codon-aware behaviour that four_fold_pi and pin_pis both rely on, so the
two analyses cannot silently diverge again."""
import math

from stats import _codon_diversity as cd


def _approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) < tol


def test_fourfold_prefixes():
    # The eight classic 4-fold-degenerate families (Leu/Val/Ser/Pro/Thr/Ala/Arg/Gly).
    assert cd.FOURFOLD_PREFIXES == frozenset(
        {"CT", "GT", "TC", "CC", "AC", "GC", "CG", "GG"}
    )


def test_site_pi_basic():
    # 2 A, 1 T: n=3, 1 - (4+1)/9 = 4/9, *3/2 = 2/3.
    assert _approx(cd.site_pi("AAT"), 2.0 / 3.0)
    assert cd.site_pi("A") is None          # < 2 called
    assert cd.site_pi("ANNN") is None       # only 1 called
    assert _approx(cd.site_pi(" A T -".replace(" ", "")), 1.0)  # 'AT-' -> A,T,(-) -> 2 called, pi=1


def test_class_aware_fourfold_simple():
    # GCA / GCG : Ala, 4-fold at pos 3; third bases A,G -> pi = 1.0
    pi, n = cd.class_aware_locus_pi(["GCA", "GCG"], [(0, 2, "4")])
    assert n == 1 and _approx(pi, 1.0)


def test_class_aware_fourfold_excludes_unknown_family():
    # The documented bug: ['GCA','NNG'] must NOT yield pi at the 3rd position,
    # because 'NNG' has an unknown codon family (N at positions 1-2). Only 'GCA'
    # qualifies -> a single base -> uncallable -> no site.
    pi, n = cd.class_aware_locus_pi(["GCA", "NNG"], [(0, 2, "4")])
    assert n == 0 and math.isnan(pi)


def test_class_aware_zerofold():
    # ATG (Met) and ACG (Thr) are each 0-fold at position 2 (index 1); bases T,C -> pi=1.0
    assert cd.ZEROFOLD[("ATG", 1)] and cd.ZEROFOLD[("ACG", 1)]
    pi, n = cd.class_aware_locus_pi(["ATG", "ACG"], [(0, 1, "0")])
    assert n == 1 and _approx(pi, 1.0)


def test_class_aware_zerofold_excludes_partial_codon():
    # A haplotype whose codon is N/gap-containing must not contribute to a 0-fold site.
    pi, n = cd.class_aware_locus_pi(["ATG", "A-G"], [(0, 1, "0")])
    assert n == 0 and math.isnan(pi)


def test_classify_zero_four_combined():
    # Two Ala codons (GCx) -> position 3 (index 2) is 4-fold; positions 1,2 are 0-fold.
    cls = dict((col, c) for col, c in cd.classify_zero_four(["GCA", "GCC"], 3))
    assert cls.get(2) == "4"
    assert cls.get(0) == "0" and cls.get(1) == "0"


def test_classify_skips_disagreeing_codons():
    # GCA (Ala, pos3 4-fold) vs ATG (Met, pos3 0-fold) disagree at pos3 -> skipped.
    cls = dict((col, c) for col, c in cd.classify_zero_four(["GCA", "ATG"], 3))
    assert 2 not in cls


def test_fourfold_codon_starts_requires_all_called_agree():
    assert list(cd.fourfold_codon_starts(["GCA", "GCG"], 3)) == [0]   # both Ala
    assert list(cd.fourfold_codon_starts(["GCA", "ATG"], 3)) == []    # Met not 4-fold
    # N at positions 1-2 is ignored for classification (family unknown):
    assert list(cd.fourfold_codon_starts(["GCA", "NNA"], 3)) == [0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
