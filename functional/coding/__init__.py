"""Coding functional scoring: map orientation-differentiating CDS fixed differences to
protein consequences, score the missense subset with three orthogonal variant-effect
predictors (AlphaMissense, ESM C, Evo 2 zero-shot), and consolidate into a per-site
functional-call table.
"""
from . import combine, map_consequences, score_alphamissense  # noqa: F401

__all__ = ["combine", "map_consequences", "score_alphamissense", "score_esmc", "score_evo2"]
