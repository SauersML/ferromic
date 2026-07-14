"""Canonical featured-locus definitions for the functional-consequence analyses.

Every locus the paper features, keyed by its ferromic ``inv_id`` (``chrom:start-end``,
hg38, no ``chr`` prefix). ``inversions.tsv`` uses a ``chrN:`` prefix on the locus id;
:func:`norm` strips it so either form matches.

Tags:
  ``phewas``       one of the PheWAS / mechanism-discussion loci (10q22, 6q27, 12q13, 17q21)
  ``selection``    a locus with an allele-frequency selection trajectory in ferromic
  ``conservation`` a top differential-CDS-conservation locus (BH q<0.05)

A locus may carry several tags (17q21 carries all three).
"""
from __future__ import annotations

FEATURED: dict[str, dict] = {
    "17:45585159-46292045": dict(
        band="17q21.31", size_kb=707, tags=["phewas", "selection", "conservation"],
        note="marquee locus; H1/H2; MAPT/KANSL1/SPPL2C block",
    ),
    "10:79542901-80217413": dict(
        band="10q22.2", size_kb=675, tags=["phewas", "selection", "conservation"],
        note="SFTPA1/SFTPA2/SFTPD surfactant cluster",
    ),
    "6:167209001-167357782": dict(
        band="6q27", size_kb=149, tags=["phewas"],
        note="TTLL2/UNC93A; '6q24' in plan shorthand, cytoband 6q27",
    ),
    "12:46896694-46915975": dict(
        band="12q13.11", size_kb=19, tags=["phewas", "selection"],
        note="intergenic; no coding fixed difference; regulatory candidate",
    ),
    "8:7301024-12598379": dict(
        band="8p23.1", size_kb=5297, tags=["selection", "conservation"],
        note="classic large polymorphic inversion; recurrent",
    ),
    "7:54234014-54308393": dict(
        band="7p11.2", size_kb=74, tags=["selection"], note="selection locus",
    ),
    "7:73113989-74799029": dict(
        band="7q11.23", size_kb=1685, tags=["conservation"],
        note="Williams-Beuren region; strongest CDS-conservation differential",
    ),
    "7:65219157-65531823": dict(band="7q11.21", size_kb=313, tags=["conservation"], note="recurrent"),
    "15:30618103-32153204": dict(band="15q13.3", size_kb=1535, tags=["conservation"], note="15q13.3 microdeletion region"),
    "16:14954790-15100859": dict(band="16p13.11", size_kb=146, tags=["conservation"], note="16p13.11 CNV region"),
    "16:28471892-28637651": dict(band="16p11.2", size_kb=166, tags=["conservation"], note="16p11.2 region"),
}


def norm(inv_id: str) -> str:
    """Return a prefix-free ``C:start-end`` id (strip a leading ``chr``)."""
    s = str(inv_id)
    if s.startswith("chr"):
        s = s[3:]
    return s


def is_featured(inv_id: str) -> bool:
    return norm(inv_id) in FEATURED


def meta(inv_id: str) -> dict | None:
    return FEATURED.get(norm(inv_id))
