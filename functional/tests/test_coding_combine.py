"""Reproduction + logic tests for the coding functional calls (run in CI from committed data)."""
import os

from functional.coding import combine
from .conftest import DATA, RESULTS, read_tsv


def test_method_flags_reproduce_recorded():
    """The documented thresholds recreate the recorded per-method damaging flags and
    n_methods_flag exactly on the frozen 3-method table."""
    scored = read_tsv(os.path.join(DATA, "coding", "arm1_final_3method.tsv"))
    recombined = combine.combine_methods(scored)
    assert len(recombined) == len(scored)
    for orig, got in zip(scored, recombined):
        assert got["am_damaging"] == (orig["am_damaging"] == "True"), orig
        assert got["esmc_damaging"] == (orig["esmc_damaging"] == "True"), orig
        assert got["evo2_disruptive"] == (orig["evo2_disruptive"] == "True"), orig
        assert got["n_methods_flag"] == int(orig["n_methods_flag"]), orig


def test_consolidate_reproduces_coding_calls():
    """consolidate() reproduces the recorded coding-call table row-for-row."""
    variants = read_tsv(os.path.join(DATA, "coding", "arm1_coding_variants.tsv"))
    scores = read_tsv(os.path.join(DATA, "coding", "arm1_final_3method.tsv"))
    got = combine.consolidate(variants, scores)
    recorded = read_tsv(os.path.join(RESULTS, "coding", "arm1_coding_calls.tsv"))

    assert len(got) == len(recorded)
    by_key = {(r["gene"], r["protein_change"], r["inv_id"]): r for r in recorded}
    for g in got:
        rec = by_key[(g["gene"], g["protein_change"], g["inv_id"])]
        assert g["coding_call"] == rec["coding_call"], (g["gene"], g["protein_change"])
        assert str(g["n_methods_flag"]) == (rec["n_methods_flag"] or ""), g["gene"]


def test_sppl2c_r461p_is_three_of_three():
    """The headline call: SPPL2C R461P at 17q21 is flagged by all three methods."""
    scores = read_tsv(os.path.join(DATA, "coding", "arm1_final_3method.tsv"))
    row = next(r for r in scores if r["gene_name"] == "SPPL2C" and r["protein_change"] == "R461P")
    am_d, esmc_d, evo2_d, n = combine.method_flags(
        float(row["am_pathogenicity"]), float(row["esmc_llr"]), float(row["evo2_delta_ll"]))
    assert (am_d, esmc_d, evo2_d, n) == (True, True, True, 3)


def test_summary_counts():
    variants = read_tsv(os.path.join(DATA, "coding", "arm1_coding_variants.tsv"))
    scores = read_tsv(os.path.join(DATA, "coding", "arm1_final_3method.tsv"))
    summ = combine.summarize(combine.consolidate(variants, scores))
    assert summ["n_functional_3of3"] >= 1
    assert summ["n_cds_sites"] == len(variants)
