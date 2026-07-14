"""Reproduction test for the validated gene-localised splice formulation (needs the cached
per-inversion AlphaGenome ``.npz`` scores; skips when they are not configured).

Recomputes ``ag_top_splice_gene`` / ``ag_max_splice`` per inversion from the cached scores and
checks they match the recorded ``per_inversion_table.csv``.
"""
import math
import os

import pytest

from functional import paths
from functional.splice import formulations as F
from .conftest import DATA, read_tsv


def _npz_dir():
    try:
        return paths.resolve("alphagenome_scores")
    except FileNotFoundError:
        return None


pytestmark = pytest.mark.skipif(
    _npz_dir() is None,
    reason="AlphaGenome per-inversion .npz not configured (set FUNCTIONAL_ALPHAGENOME_SCORES / FUNCTIONAL_DATA_ROOT)",
)


def test_top_splice_gene_matches_recorded():
    ag = F.load_all(_npz_dir())
    recorded = {r["locus"]: r for r in read_tsv(os.path.join(DATA, "regulatory", "per_inversion_table.csv"))}

    checked = 0
    for locus, genes in ag.items():
        rec = recorded.get(locus)
        if rec is None or rec.get("ag_top_splice_gene", "") in ("", "None"):
            continue
        _, name, ag_max = F.top_splice_gene(genes)
        checked += 1
        assert name == rec["ag_top_splice_gene"], locus
        assert math.isclose(ag_max, float(rec["ag_max_splice"]), rel_tol=1e-2, abs_tol=1e-2), locus
    assert checked >= 5, f"only {checked} loci overlapped for comparison"
