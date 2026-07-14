"""Structure-vs-SNV in-silico decomposition of an inversion's predicted molecular effect.

Holds the linked SNVs fixed and flips only orientation (a sequence-model counterfactual
that data alone cannot run) to split an inversion's predicted splice-usage disruption into a
structural (orientation) component and a linked-SNV component. See ``README.md``.
"""
from . import decompose  # noqa: F401
