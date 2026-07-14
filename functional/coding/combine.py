"""Cross-method combination and consolidation for the coding functional calls.

Three orthogonal variant-effect predictors are run over the orientation-differentiating
missense variants (see ``score_alphamissense``, ``score_esmc``, ``score_evo2``):

  * AlphaMissense pathogenicity (+ a same-gene matched-null percentile),
  * ESM C masked-marginal log-likelihood ratio (protein language model),
  * Evo 2 7B zero-shot delta-likelihood (DNA language model).

:func:`combine_methods` turns the three per-variant scores into per-method damaging
flags and a cross-method concordance count ``n_methods_flag``. :func:`consolidate`
joins the flagged scores back onto the full consequence annotation (all CDS sites,
including synonymous and unscored missense) and assigns a human-readable
``coding_call`` per site, tagging each by featured locus.

Thresholds
----------
* AlphaMissense: ``am_pathogenicity >= 0.564`` — the published likely-pathogenic cutoff
  (Cheng et al., Science 2023).
* Evo 2 zero-shot: ``evo2_delta_ll <= -10.0`` — disruptive-effect cutoff.
* ESM C: ``esmc_llr <= -5.0`` — deleterious-effect cutoff.

These reproduce the recorded per-method flags and ``n_methods_flag`` on the frozen
3-method table exactly (see ``functional/tests/test_coding_combine.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import featured_loci as FL

AM_PATHOGENIC = 0.564
EVO2_DISRUPTIVE = -10.0
ESMC_DELETERIOUS = -5.0

# coding_call priority for sorting (lower = stronger)
_CALL_RANK = {
    "functional (3/3 methods)": 0,
    "likely functional (2/3 methods)": 1,
    "sequence-model-only (1/3)": 2,
    "benign (0/3 methods)": 3,
    "missense, unscored": 4,
    "synonymous (no protein change)": 5,
}


@dataclass(frozen=True)
class MethodThresholds:
    am_pathogenic: float = AM_PATHOGENIC
    evo2_disruptive: float = EVO2_DISRUPTIVE
    esmc_deleterious: float = ESMC_DELETERIOUS


def _to_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def method_flags(am: float | None, esmc: float | None, evo2: float | None,
                 t: MethodThresholds = MethodThresholds()) -> tuple[bool, bool, bool, int]:
    """Return ``(am_damaging, esmc_damaging, evo2_disruptive, n_methods_flag)``.

    A missing score for a method counts as not-flagged by that method.
    """
    am_dmg = am is not None and am >= t.am_pathogenic
    esmc_dmg = esmc is not None and esmc <= t.esmc_deleterious
    evo2_dis = evo2 is not None and evo2 <= t.evo2_disruptive
    return am_dmg, esmc_dmg, evo2_dis, int(am_dmg) + int(esmc_dmg) + int(evo2_dis)


def combine_methods(scored_rows: list[dict], t: MethodThresholds = MethodThresholds()) -> list[dict]:
    """Given per-variant rows with ``am_pathogenicity`` / ``esmc_llr`` / ``evo2_delta_ll``,
    attach ``am_damaging`` / ``esmc_damaging`` / ``evo2_disruptive`` / ``n_methods_flag``.
    Returns new dicts; inputs are not mutated."""
    out = []
    for r in scored_rows:
        am = _to_float(r.get("am_pathogenicity"))
        esmc = _to_float(r.get("esmc_llr"))
        evo2 = _to_float(r.get("evo2_delta_ll"))
        am_dmg, esmc_dmg, evo2_dis, n = method_flags(am, esmc, evo2, t)
        rec = dict(r)
        rec.update(am_damaging=am_dmg, esmc_damaging=esmc_dmg,
                   evo2_disruptive=evo2_dis, n_methods_flag=n)
        out.append(rec)
    return out


def _coding_call(consequence: str, n_flag: int | None) -> str:
    if consequence != "missense":
        return "synonymous (no protein change)"
    if n_flag is None:
        return "missense, unscored"
    if n_flag >= 3:
        return "functional (3/3 methods)"
    if n_flag == 2:
        return "likely functional (2/3 methods)"
    if n_flag == 1:
        return "sequence-model-only (1/3)"
    return "benign (0/3 methods)"


def consolidate(variants: list[dict], scores: list[dict]) -> list[dict]:
    """Join the full consequence annotation (``variants``: one row per CDS site) to the
    3-method ``scores`` (missense subset, keyed by ``(gene_name, protein_change)``) and
    assign a ``coding_call`` per site. Mirrors the supplement's coding table.
    """
    skey = {(s["gene_name"], s["protein_change"]): s for s in scores}
    out = []
    for v in variants:
        gene = v["gene_name"]
        inv = FL.norm(v["inv_id"])
        cons = v["consequence"]
        pc = v.get("protein_change", "")
        rec = {
            "inv_id": inv,
            "band": (FL.meta(inv) or {}).get("band", ""),
            "is_featured": FL.is_featured(inv),
            "gene": gene,
            "transcript": v.get("transcript_id", ""),
            "cds_pos": v.get("cds_pos_1based", ""),
            "consequence": cons,
            "protein_change": pc,
            "aa_ref": v.get("aa_ref", ""),
            "aa_alt": v.get("aa_alt", ""),
            "g_pos_hg38": v.get("g_pos_1based", ""),
            "am_pathogenicity": "", "am_percentile_in_gene": "",
            "esmc_llr": "", "evo2_delta_ll": "", "n_methods_flag": "", "coding_call": "",
        }
        s = skey.get((gene, pc)) if cons == "missense" else None
        if s is not None:
            rec["am_pathogenicity"] = s.get("am_pathogenicity", "")
            rec["am_percentile_in_gene"] = s.get("am_percentile_in_gene", "")
            rec["esmc_llr"] = s.get("esmc_llr", "")
            rec["evo2_delta_ll"] = s.get("evo2_delta_ll", "")
            n = s.get("n_methods_flag")
            rec["n_methods_flag"] = n
            rec["coding_call"] = _coding_call(cons, int(n) if n not in (None, "") else None)
        else:
            rec["coding_call"] = _coding_call(cons, None)
        out.append(rec)

    out.sort(key=lambda r: (not r["is_featured"], _CALL_RANK.get(r["coding_call"], 9), r["inv_id"]))
    return out


def summarize(rows: list[dict]) -> dict:
    """Summary counts over a consolidated coding-call table."""
    def n_call(prefix):
        return sum(1 for r in rows if r["coding_call"].startswith(prefix))
    return dict(
        n_cds_sites=len(rows),
        n_missense=sum(1 for r in rows if r["consequence"] == "missense"),
        n_synonymous=sum(1 for r in rows if r["consequence"] == "synonymous"),
        n_genes=len({r["gene"] for r in rows}),
        n_loci=len({r["inv_id"] for r in rows}),
        n_functional_3of3=n_call("functional"),
        n_likely_2of3=n_call("likely"),
        n_seqmodel_only=n_call("sequence-model"),
        top_call=[dict(gene=r["gene"], pc=r["protein_change"], band=r["band"],
                       am=r["am_pathogenicity"], esmc=r["esmc_llr"],
                       evo2=r["evo2_delta_ll"], call=r["coding_call"])
                  for r in rows if r["coding_call"].startswith(("functional", "likely"))],
    )
