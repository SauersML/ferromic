"""Assemble the per-locus master table, QC summary, and de-biased summary from the
per-window decomposition results — pure post-processing, no API, no randomness.

Inputs (a results directory, default = the committed ``functional/results/structural``):
  ag_decomp_full.json     per-window consensus decomposition norms (structural anchor)
  bg_stats.json           background genetics (divergence, LD/tagging)
  structural_anchor.json  breakpoint-in-gene structural anchor
  locus_verdicts.json     per-locus verdicts
  handle3_conditional.json 17q21 conditional-collapse readout
  perhap_debiased.json    de-biased per-haplotype decomposition (headline)

Outputs (written to the results directory):
  master_table.csv, qc_summary.json, debias_summary.json

``fraction_structural`` and the de-biased fraction are computed by
:mod:`functional.structural.decompose` — the same functions the reproduction test checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

try:
    from . import decompose as D
    from . import _data
except ImportError:  # loose script
    import decompose as D  # type: ignore
    import _data  # type: ignore

TAG_RELIABLE_MIN = 0.90       # tag-Porubsky concordance gate for primary claims
STRUCTURE_DOMINANT_MIN = 0.66  # fraction_structural threshold for "structure-dominant"


def _load(results_dir, name):
    with open(os.path.join(results_dir, name)) as fh:
        return json.load(fh)


def build_master(results_dir):
    ag = {r["locus"]: r for r in _load(results_dir, "ag_decomp_full.json") if "windows" in r}
    bg = {r["locus"]: r for r in _load(results_dir, "bg_stats.json") if "error" not in r}
    anchor = {r["locus"]: r for r in _load(results_dir, "structural_anchor.json")}
    verd = {r["locus"]: r for r in _load(results_dir, "locus_verdicts.json")["rows"]}
    try:
        h3 = {r["locus"]: r for r in _load(results_dir, "handle3_conditional.json")}
    except FileNotFoundError:
        h3 = {}

    rows = []
    for loc, x in ag.items():
        b = bg.get(loc, {})
        a = anchor.get(loc, {})
        v = verd.get(loc, {})
        c = h3.get(loc, {})
        fs = D.locus_fraction_structural(x["windows"])
        bgd = x["background"]
        n_hi = bgd["n_hap_inverted"]
        n_hd = bgd["n_hap_direct"]
        conc = bgd["tag_porubsky_concordance"]
        rows.append({
            "locus": loc, "chrom": x["chrom"], "size_bp": x["size"],
            "recurrent": {"1": "recurrent", "0": "single"}.get(x["recur_consensus"], "unknown"),
            "inv_AF": x["inv_AF"], "n_hap_inverted": n_hi, "n_hap_direct": n_hd,
            "tag_porubsky_concordance": round(conc, 3),
            "tag_reliable": bool(conc >= TAG_RELIABLE_MIN),
            "fraction_structural": round(fs, 3) if math.isfinite(fs) else None,
            "between_bg_divergence": round(b.get("between_bg_divergence", float("nan")), 3)
            if b.get("between_bg_divergence") is not None else None,
            "n_diff_sites_050": b.get("n_diff_sites_050"),
            "tag_max_r2_cis": b.get("tag_max_r2_cis"),
            "breakpoint_in_gene": a.get("breakpoint_in_gene"),
            "bp_mediated_genes": ";".join(a.get("bp_mediated_genes", [])),
            "measured_any": b.get("measured_any"),
            "cond_marginal_p": c.get("marginal_p"), "cond_inv_p": c.get("conditional_inv_p"),
            "cond_top_cis_r2_with_inv": c.get("top_cis_r2_with_inversion"),
            "verdict": v.get("verdict"),
        })
    rows.sort(key=lambda r: (-(r["measured_any"] or 0), -(r["fraction_structural"] or 0)))
    return rows


def build_qc(rows, results_dir):
    ag_raw = _load(results_dir, "ag_decomp_full.json")
    ag = {r["locus"]: r for r in ag_raw if "windows" in r}
    qc = {"reconciliation_max_abs_err": 0.0, "rc_roundtrip_all_ok": True, "len_all_ok": True,
          "total_variant_refmismatch": 0, "loci_scored": len(ag), "hap_af_check": []}
    for loc, x in ag.items():
        for w in x["windows"]:
            qc["reconciliation_max_abs_err"] = max(
                qc["reconciliation_max_abs_err"], w["reconciliation_max_abs_err"])
            qc["rc_roundtrip_all_ok"] &= w["qc"]["rc_roundtrip_ok"]
            qc["len_all_ok"] &= w["qc"]["len_ok"]
            qc["total_variant_refmismatch"] += (
                w["qc"]["n_inv_variants_refmismatch"] + w["qc"]["n_dir_variants_refmismatch"])
        bgd = x["background"]
        n_hi, n_hd = bgd["n_hap_inverted"], bgd["n_hap_direct"]
        hap_af = n_hi / (n_hi + n_hd) if (n_hi + n_hd) else float("nan")
        try:
            af_stated = float(x["inv_AF"])
            qc["hap_af_check"].append({"locus": loc, "tag_hap_AF": round(hap_af, 3),
                                       "stated_inv_AF": af_stated,
                                       "abs_diff": round(abs(hap_af - af_stated), 3)})
        except (TypeError, ValueError):
            pass
    reliable = [r for r in rows if r["tag_reliable"]]
    all_fs = [r["fraction_structural"] for r in rows]
    rel_fs = [r["fraction_structural"] for r in reliable]
    r2s = [r["tag_max_r2_cis"] for r in rows if r["tag_max_r2_cis"] is not None]
    qc["hap_af_max_abs_diff"] = max((h["abs_diff"] for h in qc["hap_af_check"]), default=None)
    qc["n_tag_reliable"] = len(reliable)
    qc["median_fraction_structural_all"] = round(D.median(all_fs), 3)
    qc["median_fraction_structural_reliable"] = round(D.median(rel_fs), 3)
    qc["n_structure_dominant_reliable"] = sum(
        1 for r in reliable if (r["fraction_structural"] or 0) >= STRUCTURE_DOMINANT_MIN)
    qc["median_fraction_structural"] = qc["median_fraction_structural_reliable"]
    qc["median_tag_max_r2"] = round(D.median(r2s), 3)
    return qc


def build_debias(results_dir):
    perhap = _load(results_dir, "perhap_debiased.json")
    consensus = {r["locus"]: r for r in _load(results_dir, "ag_decomp_full.json") if "windows" in r}
    locus_debiased, locus_consensus, indiv_min, indiv_max = [], [], [], []
    for rec in perhap:
        fr = D.locus_fraction_debiased(rec["windows"])
        locus_debiased.append(fr)
        cons = consensus.get(rec["locus"])
        if cons:
            locus_consensus.append(D.locus_fraction_structural(cons["windows"]))
        for w in rec["windows"]:
            if "fraction_individual_min" in w:
                indiv_min.append(w["fraction_individual_min"])
                indiv_max.append(w["fraction_individual_max"])
    return {
        "median_consensus": round(D.median(locus_consensus), 3),
        "median_debiased": D.median(locus_debiased),
        "n": len(perhap),
        "n_structure_dominant": sum(1 for f in locus_debiased if f >= 0.5),
        "locus_min_debiased": min(locus_debiased) if locus_debiased else float("nan"),
        "locus_max_debiased": max(locus_debiased) if locus_debiased else float("nan"),
        "indiv_min": min(indiv_min) if indiv_min else float("nan"),
        "indiv_max": max(indiv_max) if indiv_max else float("nan"),
    }


def run(results_dir, write=True):
    rows = build_master(results_dir)
    qc = build_qc(rows, results_dir)
    debias = build_debias(results_dir)
    if write:
        with open(os.path.join(results_dir, "master_table.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        json.dump(qc, open(os.path.join(results_dir, "qc_summary.json"), "w"), indent=1)
        json.dump(debias, open(os.path.join(results_dir, "debias_summary.json"), "w"), indent=1)
    return rows, qc, debias


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble structural master table + QC + de-biased summary")
    ap.add_argument("--results-dir", default=_data.RESULTS_DIR,
                    help="directory with the per-window decomposition JSONs (default: committed results/structural)")
    ap.add_argument("--no-write", action="store_true", help="compute + print only")
    a = ap.parse_args(argv)
    rows, qc, debias = run(os.path.abspath(a.results_dir), write=not a.no_write)
    print(f"loci: {len(rows)}  tag-reliable: {qc['n_tag_reliable']}  "
          f"median fraction_structural (reliable): {qc['median_fraction_structural_reliable']}  "
          f"structure-dominant: {qc['n_structure_dominant_reliable']}/{qc['n_tag_reliable']}")
    print(f"de-biased median: {debias['median_debiased']:.3f} "
          f"(consensus upper bound {debias['median_consensus']:.3f}; n={debias['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
