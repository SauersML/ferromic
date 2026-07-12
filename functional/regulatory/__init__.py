"""Measured regulatory-QTL analyses at inversions in real RNA-seq (haplotype-level).

* :mod:`eqtl`          Geuvadis cis-eQTL by inversion-tag dosage.
* :mod:`sqtl_geuvadis` Geuvadis cis differential splicing (junction / exon / transcript).
* :mod:`sqtl_gtex`     GTEx v10 multi-tissue sQTL lookup + MAF/proximity-matched enrichment.
* :mod:`integrate`     Per-locus integration of measured + predicted regulatory consequences.
"""
from . import common, eqtl, integrate, sqtl_geuvadis  # noqa: F401

__all__ = ["common", "eqtl", "sqtl_geuvadis", "sqtl_gtex", "integrate"]
