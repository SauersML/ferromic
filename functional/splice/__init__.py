"""Gene-localised AlphaGenome splice-disruption scoring and its GTEx-sQTL validation.

* :mod:`score_alphagenome` AlphaGenome API driver (per-inversion .npz; network + API key).
* :mod:`formulations`      the validated gene-localised splice formulation (from cached .npz).
* :mod:`validate_gtex`     top-splice gene vs measured GTEx sGenes.
"""
from . import formulations, validate_gtex  # noqa: F401

__all__ = ["score_alphagenome", "formulations", "validate_gtex"]
