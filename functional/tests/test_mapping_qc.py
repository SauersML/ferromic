"""CDS -> residue mapping QC: codon assembly, strand handling, reverse-complement round-trip.

These exercise the silent-bug guards in :mod:`functional.coding.map_consequences` on synthetic
sequence, so they run without the reference genome/GTF.
"""
from functional.coding import codons
from functional.coding.map_consequences import cds_genomic_positions, map_variant


class FakeFasta:
    """Minimal pyfaidx.Fasta stand-in: fa[chrom][i].seq returns the 1-char base at 0-based i."""
    class _Base:
        def __init__(self, s):
            self.seq = s

    class _Chrom:
        def __init__(self, seq):
            self.seq = seq

        def __getitem__(self, i):
            return FakeFasta._Base(self.seq[i])

    def __init__(self, seqs):
        self._seqs = seqs

    def __getitem__(self, chrom):
        return FakeFasta._Chrom(self._seqs[chrom])


def test_revcomp_round_trip():
    for s in ["ACGT", "AACCGGTT", "N", "ACGTACGTACGT"]:
        assert codons.revcomp(codons.revcomp(s)) == s
    assert codons.revcomp("ACGT") == "ACGT"[::-1].translate(codons.COMP)


def test_translate_known_codons():
    assert codons.translate("ATGGCC") == "MA"
    assert codons.translate("TAA") == "*"
    assert codons.classify_consequence("A", "A") == "synonymous"
    assert codons.classify_consequence("R", "P") == "missense"
    assert codons.classify_consequence("W", "*") == "stop_gained"


def test_plus_strand_missense_mapping():
    # chrom seq (0-based): positions 10..12 -> codon "CGT" (=R). Substitute base 3 (C->A at cds1)
    seq = "N" * 10 + "CGT" + "N" * 10
    fa = FakeFasta({"chr1": seq})
    rec = {"chrom": "chr1", "strand": "+", "cds": [(11, 13)]}  # 1-based genomic 11..13
    res = map_variant(rec, cds_pos_1based=1, direct_allele="C", inverted_allele="A", fa=fa)
    assert res["status"] == "OK"
    assert res["ref_match"] is True
    assert res["g_pos_1based"] == 11
    assert res["codon_ref"] == "CGT" and res["aa_ref"] == "R"
    assert res["codon_alt"] == "AGT" and res["aa_alt"] == "S"
    assert res["consequence"] == "missense"
    assert res["g_ref"] == "C" and res["g_alt"] == "A"


def test_minus_strand_uses_reverse_complement():
    # transcript on '-' strand: genomic 11..13 = "ACG"; transcript sense (revcomp) = "CGT" (R)
    seq = "N" * 10 + "ACG" + "N" * 10
    fa = FakeFasta({"chr1": seq})
    rec = {"chrom": "chr1", "strand": "-", "cds": [(11, 13)]}
    pos, chrom, strand = cds_genomic_positions(rec)
    assert pos[0] == 13 and strand == "-"  # 5'->3' on minus strand starts at the higher coord
    res = map_variant(rec, cds_pos_1based=1, direct_allele="C", inverted_allele="A", fa=fa)
    assert res["status"] == "OK"
    # cds position 1 = genomic 13 (G on plus) -> transcript-sense base C
    assert res["g_pos_1based"] == 13
    assert res["ref_match"] is True
    # plus-strand g_alt is revcomp of the transcript-sense alt allele "A" -> "T"
    assert res["g_alt"] == "T"


def test_ref_mismatch_is_flagged_not_dropped():
    seq = "N" * 10 + "CGT" + "N" * 10
    fa = FakeFasta({"chr1": seq})
    rec = {"chrom": "chr1", "strand": "+", "cds": [(11, 13)]}
    res = map_variant(rec, cds_pos_1based=1, direct_allele="G", inverted_allele="A", fa=fa)
    assert res["status"] == "OK"
    assert res["ref_match"] is False  # reported, caller excludes from scoring
