"""Map orientation-differentiating CDS fixed differences to genomic coordinates and
amino-acid consequences, using the GENCODE r47 gene model + GRCh38.

Input:  ferromic ``gene_inversion_fixed_differences.tsv`` (gene, transcript, CDS position,
        direct/inverted allele).
Output: a per-variant table with genomic coordinate, strand, plus-strand ref/alt, codon
        change, ``aa_ref``/``aa_alt``, ``consequence``, and a REF-match verification flag.

The REF-match flag is the silent-bug guard: the ferromic table's CDS positions were
computed on a possibly different transcript version, so we reconstruct the CDS from the
r47 model and require ``direct_allele`` to equal the reference CDS base at ``cds_pos``.
Rows that fail are flagged (``ref_match=False``) and excluded from scoring — reported,
not silently dropped.
"""
from __future__ import annotations

from collections import defaultdict

from .codons import CODON_TABLE, classify_consequence, revcomp


def parse_gtf_cds(gtf_path: str, wanted_tx) -> dict:
    """``{tx_base: {'strand','chrom','cds':[(start,end),...] sorted}}`` for the wanted
    (unversioned) transcript ids."""
    import gzip
    wanted = {t.split(".")[0] for t in wanted_tx}
    out: dict = defaultdict(lambda: {"strand": None, "chrom": None, "cds": []})
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "CDS":
                continue
            idx = f[8].find('transcript_id "')
            if idx < 0:
                continue
            tbase = f[8][idx + 15:].split('"', 1)[0].split(".")[0]
            if tbase not in wanted:
                continue
            rec = out[tbase]
            rec["strand"], rec["chrom"] = f[6], f[0]
            rec["cds"].append((int(f[3]), int(f[4])))
    for rec in out.values():
        rec["cds"].sort()
    return dict(out)


def cds_genomic_positions(rec: dict) -> tuple[list[int], str, str]:
    """Genomic 1-based positions in transcript 5'->3' order, plus chrom and strand."""
    pos: list[int] = []
    for s, e in rec["cds"]:
        pos.extend(range(s, e + 1))
    if rec["strand"] == "-":
        pos = pos[::-1]
    return pos, rec["chrom"], rec["strand"]


def map_variant(rec: dict, cds_pos_1based: int, direct_allele: str, inverted_allele: str, fa) -> dict:
    """Map one CDS fixed difference to its genomic + protein consequence.

    ``fa`` is a pyfaidx.Fasta-like object indexable as ``fa[chrom][i].seq``. Returns a dict
    with ``status`` = OK / TRANSCRIPT_NOT_IN_GTF / CDS_POS_OUT_OF_RANGE / INCOMPLETE_CODON.
    """
    if not rec or not rec["cds"]:
        return {"status": "TRANSCRIPT_NOT_IN_GTF"}
    pos, chrom, strand = cds_genomic_positions(rec)
    cp = int(cds_pos_1based)
    if cp < 1 or cp > len(pos):
        return {"status": "CDS_POS_OUT_OF_RANGE", "cds_len": len(pos), "chrom": chrom, "strand": strand}
    gpos = pos[cp - 1]
    ref_genomic = fa[chrom][gpos - 1].seq.upper()
    ref_cds_base = ref_genomic if strand == "+" else revcomp(ref_genomic)
    ref_match = ref_cds_base == direct_allele

    codon_idx = (cp - 1) // 3
    codon_offset = (cp - 1) % 3
    codon_pos = pos[codon_idx * 3: codon_idx * 3 + 3]
    if len(codon_pos) < 3:
        return {"status": "INCOMPLETE_CODON", "chrom": chrom, "strand": strand, "g_pos_1based": gpos,
                "ref_match": ref_match}
    codon_ref = "".join(
        (fa[chrom][p - 1].seq.upper() if strand == "+" else revcomp(fa[chrom][p - 1].seq.upper()))
        for p in codon_pos
    )
    codon_alt = codon_ref[:codon_offset] + inverted_allele + codon_ref[codon_offset + 1:]
    aa_ref = CODON_TABLE.get(codon_ref, "?")
    aa_alt = CODON_TABLE.get(codon_alt, "?")
    aa_num = codon_idx + 1
    return {
        "status": "OK", "chrom": chrom, "strand": strand, "g_pos_1based": gpos,
        "g_ref": ref_genomic, "g_alt": inverted_allele if strand == "+" else revcomp(inverted_allele),
        "ref_match": ref_match, "codon_ref": codon_ref, "codon_alt": codon_alt,
        "aa_ref": aa_ref, "aa_alt": aa_alt, "aa_num": aa_num,
        "protein_change": f"{aa_ref}{aa_num}{aa_alt}",
        "consequence": classify_consequence(aa_ref, aa_alt),
    }
