"""Pure decomposition arithmetic for the structure-vs-SNV split.

No I/O, no API, no randomness — just the flank-restricted L1 bookkeeping that turns
per-window disruption norms into a per-locus ``fraction_structural``. Both the summary
assembly (:mod:`functional.structural.summarize`) and the reproduction test import these
functions, so the recorded tables and the CI check share one definition.

Decomposition identity (see ``score_consensus.py`` / the module README):

    d_total  = SSU(full_inverted) - SSU(full_direct)
    d_struct = SSU(ref_inverted)  - SSU(ref_direct)          # pure orientation flip
    d_snv    = d_total - d_struct                            # differential linked SNVs

Disruption magnitudes ``D_struct`` / ``D_snv`` are L1 norms of these fields restricted to
positions OUTSIDE the flipped segment ("flank"). Per locus the fractions are combined
disruption-weighted across windows:

    fraction_structural = sum_w D_struct_w / sum_w (D_struct_w + D_snv_w)
"""
from __future__ import annotations

import math
from typing import Iterable


def locus_fraction_structural(windows: Iterable[dict]) -> float:
    """Disruption-weighted structural fraction over a locus's consensus windows.

    Each window dict must carry ``disruption_flank`` with ``struct`` and ``snv`` L1 norms
    (the shape written by ``score_consensus.py``). Matches experiment #10's ``frac_struct``.
    """
    num = den = 0.0
    for w in windows:
        s = w["disruption_flank"]["struct"]
        n = w["disruption_flank"]["snv"]
        num += s
        den += s + n
    return num / den if den > 0 else float("nan")


def locus_fraction_debiased(windows: Iterable[dict]) -> float:
    """Disruption-weighted DE-BIASED structural fraction over per-haplotype windows.

    Each window dict must carry ``D_struct`` and ``D_snv_debiased`` (the shape written by
    ``score_perhap.py``). Matches experiment #10's locus-level de-biased aggregation.
    """
    num = den = 0.0
    for w in windows:
        num += w["D_struct"]
        den += w["D_struct"] + w["D_snv_debiased"]
    return num / den if den > 0 else float("nan")


def median(values: Iterable[float]) -> float:
    """Median of the finite values (numpy-free so the test has no heavy deps)."""
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    if not xs:
        return float("nan")
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])
