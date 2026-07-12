"""Reproduction test for the Geuvadis cis-eQTL scan (needs the Geuvadis genotypes + expression
matrix; skips when they are not configured).

Runs the 17q21 smoke locus at seed 42 and checks the recomputed per-gene betas / BH-q match
the recorded ``arm_eqtl.tsv`` within tolerance.
"""
import math
import os

import pytest

from functional import paths
from functional.regulatory import eqtl
from .conftest import DATA, RESULTS, read_tsv

SMOKE_LOCUS = "chr17:45585159-46292045"


def _inputs_available():
    try:
        paths.resolve("geuvadis_gene_rpkm")
        paths.resolve("geuvadis_pgen")
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _inputs_available(),
    reason="Geuvadis genotypes/expression not configured (set FUNCTIONAL_DATA_ROOT / FUNCTIONAL_* env vars)",
)


def test_eqtl_smoke_locus_matches_recorded():
    panel = os.path.join(DATA, "1kg_panel.tsv")
    rows = eqtl.run_eqtl(
        inversions_tsv=os.path.join(DATA, "inversions.tsv"),
        gene_rpkm_path=paths.resolve("geuvadis_gene_rpkm"),
        pgen_prefix=paths.resolve("geuvadis_pgen"),
        panel_tsv=panel,
        loci_subset=[SMOKE_LOCUS],
        n_perm=50, seed=42,
    )
    assert rows, "no eQTL tests produced for the smoke locus"
    got = {r["gene_id"]: r for r in rows}

    recorded = [r for r in read_tsv(os.path.join(RESULTS, "regulatory", "arm_eqtl.tsv"))
                if r["locus"] == SMOKE_LOCUS]
    assert recorded, "smoke locus absent from recorded arm_eqtl.tsv"

    checked = 0
    for rec in recorded:
        g = got.get(rec["gene_id"])
        if g is None:
            continue
        checked += 1
        assert math.isclose(g["beta_log2fc_per_alt"], float(rec["beta_log2fc_per_alt"]),
                            rel_tol=1e-3, abs_tol=1e-3), rec["gene_sym"]
        # analytic p is deterministic; permutation p may vary with n_perm, so is not compared
        assert math.isclose(g["p"], float(rec["p"]), rel_tol=1e-3, abs_tol=1e-6), rec["gene_sym"]
    assert checked >= 5, f"only {checked} genes overlapped for comparison"
