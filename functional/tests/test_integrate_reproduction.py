"""Reproduction test for the per-locus regulatory integration (runs in CI from committed data).

integrate() joins the measured eQTL (committed recorded table), measured GTEx sQTL / Geuvadis
splicing (#8 master), measured GTEx eQTL, and AlphaGenome predicted splice into the per-locus
table. This reproduces the recorded ``regulatory_per_locus.tsv`` from committed inputs.
"""
import os

from functional.regulatory import integrate
from .conftest import DATA, RESULTS, read_tsv

_KEY_COLS = ["band", "is_featured", "eqtl_n_sig", "eqtl_top_gene", "eqtl_top_dir",
             "gtex_n_sqtl_genes", "measured_sqtl_any", "gtex_eqtl_n_genes",
             "ag_top_splice_gene", "measured_molecular_any"]


def _reproduce():
    return integrate.integrate(
        eqtl_tsv=os.path.join(RESULTS, "regulatory", "arm_eqtl.tsv"),
        sqtl_master_tsv=os.path.join(DATA, "regulatory", "per_inversion_master.tsv"),
        gtex_eqtl_tsv=os.path.join(DATA, "regulatory", "gtex_eqtls.tsv"),
        ag_splice_tsv=os.path.join(DATA, "regulatory", "per_inversion_table.csv"),
        inversions_tsv=os.path.join(DATA, "inversions.tsv"),
        ensg_symbol_tsv=os.path.join(DATA, "ensg_symbol.tsv.gz"),
    )


def test_integrate_reproduces_recorded_table():
    got = _reproduce()
    recorded = {r["locus"]: r for r in read_tsv(os.path.join(RESULTS, "regulatory", "regulatory_per_locus.tsv"))}
    assert len(got) == len(recorded)
    for g in got:
        rec = recorded[g["locus"]]
        for col in _KEY_COLS:
            gv = str(g[col]).replace("True", "1").replace("False", "0") if isinstance(g[col], bool) else str(g[col])
            rv = str(rec[col]).replace("True", "1").replace("False", "0")
            assert gv == rv, f"{g['locus']} col={col}: reproduced {gv!r} != recorded {rv!r}"


def test_summary_consistency():
    rows = _reproduce()
    summ = integrate.summarize(rows)
    assert summ["n_loci"] == len(rows)
    assert 0 <= summ["frac_measured_molecular"] <= 1
    assert summ["n_featured"] >= 1
