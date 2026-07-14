"""Validate the AlphaGenome top-splice gene against measured GTEx sQTLs.

The validation for the gene-localised splice formulation (:mod:`functional.splice.formulations`)
is: how often is the inversion's AlphaGenome-predicted top-splice gene actually a *measured*
GTEx sGene (an sQTL at the inversion's tag SNP)? A high rate means the prediction localises
disruption to the gene that measurably changes splicing, not just to any gene in the window.

``measured_sgenes`` is ``{locus: set(ensg)}`` of measured sGenes per inversion (from the GTEx
sQTL lookup, :mod:`functional.regulatory.sqtl_gtex`).
"""
from __future__ import annotations

import numpy as np

from . import formulations as F


def per_inversion_validation(ag_by_region: dict, measured_sgenes: dict) -> tuple[list[dict], dict]:
    """For each region, compute the top-splice gene and whether it is a measured sGene.

    Returns ``(rows, summary)``. ``rows`` carry ``locus``, ``ag_top_splice_gene``,
    ``ag_max_splice``, ``ag_top_splice_is_measured_sgene`` (None when the locus has no measured
    sGenes to check against). ``summary`` reports the sGene hit-rate over checkable loci.
    """
    rows = []
    hits = []
    for locus, genes in ag_by_region.items():
        gid, name, ag_max = F.top_splice_gene(genes)
        sgenes = measured_sgenes.get(locus, set())
        is_sgene = (gid in sgenes) if (gid and sgenes) else None
        if is_sgene is not None:
            hits.append(1 if is_sgene else 0)
        rows.append({
            "locus": locus,
            "ag_top_splice_gene": name,
            "ag_max_splice": round(ag_max, 3) if not np.isnan(ag_max) else None,
            "ag_top_splice_is_measured_sgene": is_sgene,
        })
    summary = {
        "n_regions": len(rows),
        "ag_topsplice_is_sgene_rate": float(np.mean(hits)) if hits else None,
        "ag_topsplice_is_sgene_n": len(hits),
    }
    return rows, summary
