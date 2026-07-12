"""Reproduction + logic tests for the structure-vs-SNV decomposition.

Pure-logic, run in CI from the committed per-window decomposition JSONs (no AlphaGenome API,
no large inputs). They check that the packaged code reproduces the recorded structural tables:
the per-locus ``fraction_structural``, the tag-reliable median (consensus upper bound), and the
de-biased headline fraction.
"""
import csv
import json
import math
import os

from functional.structural import decompose, summarize
from .conftest import RESULTS

SR = os.path.join(RESULTS, "structural")


def _read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _load(name):
    with open(os.path.join(SR, name)) as fh:
        return json.load(fh)


def test_fraction_structural_reproduces_master_table():
    """decompose.locus_fraction_structural recomputes every locus's recorded fraction_structural
    from the committed per-window norms, matching master_table.csv."""
    ag = {r["locus"]: r for r in _load("ag_decomp_full.json") if "windows" in r}
    recorded = {r["locus"]: r for r in _read_csv(os.path.join(SR, "master_table.csv"))}
    assert set(ag) == set(recorded), "locus set drift between decomposition and master table"
    for loc, x in ag.items():
        got = decompose.locus_fraction_structural(x["windows"])
        exp = float(recorded[loc]["fraction_structural"])
        assert abs(round(got, 3) - exp) <= 1e-9, (loc, got, exp)


def test_summarize_reproduces_qc_and_debias():
    """summarize.run (no write) reproduces the recorded QC summary + de-biased summary."""
    rows, qc, debias = summarize.run(SR, write=False)
    rec_qc = _load("qc_summary.json")
    rec_db = _load("debias_summary.json")

    assert qc["n_tag_reliable"] == rec_qc["n_tag_reliable"]
    assert qc["median_fraction_structural_reliable"] == rec_qc["median_fraction_structural_reliable"]
    assert qc["n_structure_dominant_reliable"] == rec_qc["n_structure_dominant_reliable"]
    assert qc["loci_scored"] == rec_qc["loci_scored"]

    assert debias["n"] == rec_db["n"]
    assert abs(debias["median_debiased"] - rec_db["median_debiased"]) <= 1e-6
    assert abs(debias["median_consensus"] - rec_db["median_consensus"]) <= 1e-6


def test_tag_reliable_median_is_structure_dominant():
    """The headline: over the 9 tag-reliable loci the consensus structural fraction is
    structure-dominant (median ~0.85, all 9 >= 0.66)."""
    rows, qc, debias = summarize.run(SR, write=False)
    assert qc["n_tag_reliable"] == 9
    assert qc["n_structure_dominant_reliable"] == qc["n_tag_reliable"]
    assert qc["median_fraction_structural_reliable"] >= 0.66


def test_debiased_below_consensus_upper_bound():
    """The de-biased headline fraction (true per-haplotype SNV load) sits below the consensus
    upper bound, and stays structure-dominant (median ~0.80)."""
    _, _, debias = summarize.run(SR, write=False)
    assert debias["median_debiased"] <= debias["median_consensus"] + 1e-9
    assert debias["median_debiased"] >= 0.5
    assert math.isclose(round(debias["median_debiased"], 2), 0.80, abs_tol=0.02)
