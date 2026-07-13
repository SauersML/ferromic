"""Validated pop-gen recurrence classifier for polymorphic inversions.

Per-inversion population-genetic features (haplotype-tree parsimony, tag-SNP r^2,
nucleotide diversity, differentiation, pairwise-distance structure) feed a
deterministic logistic classifier with a partial-AUC-at-low-FPR refinement. The
classifier is fit and validated against structured-coalescent simulation ground truth
and applied to the balanced inversion set to emit a continuous recurrence score per
inversion. Analysis code + recorded result tables only; see recurrence/README.md.
"""
from __future__ import annotations

from . import apply, classifier, features, fit, parsimony, transferable  # noqa: F401

__all__ = ["classifier", "features", "parsimony", "transferable", "fit", "apply"]
