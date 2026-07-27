#!/usr/bin/env python
"""Non-circular recurrence classification from flanking SD architecture.

Reviewer 1 objected that comparing diversity between recurrent and single-event
inversions is partly circular, because the recurrence classification itself is
built from sequence variation (a haplotype tree and tagging SNPs). If the
diversity difference is an artefact of the classifier, it should disappear once
recurrence is called from something that never touches the variation data.

Flanking segmental duplication architecture is such a signal. Porubsky et al.
(2022) show recurrence status is predicted by large, high-identity flanking
inverted repeats -- a structural property of the locus, recorded in
``inv_properties.tsv`` as ``Flanking_inverted_repeat_size.kbp.`` and
``Flanking_Inverted_repeat_identity``, and computed from assembly alignments
rather than from haplotype diversity.

What this does
--------------
1. Fits ``P(recurrent) ~ log10(SD size kbp) + SD identity(%)`` on the 93
   consensus-classified loci -- architecture only, no diversity term.
2. Produces **leave-one-out cross-validated** calls, so each locus is labelled by
   a model that never saw its own consensus label, and reports agreement with the
   consensus both in-sample and out-of-sample.
3. Refits the primary Δ-logπ model (``inv_dir_recur_model.run_model_A``, HC3 SEs,
   the same quantile-derived epsilon) with the SD-derived labels substituted for
   the consensus labels.

The classifier is trained on the consensus labels, so the *mapping* from
architecture to recurrence is learned. What matters for the circularity argument
is that the resulting per-locus label is a function of architecture alone: no
locus's diversity enters its own recurrence call.

Run from ``data/`` (the primary model resolves ``./output.csv``):

    cd data && python ../stats/recurrence_sd_architecture.py

Outputs (in ``data/``):
  recurrence_sd_calls.tsv       per-locus SD calls vs consensus
  recurrence_sd_summary.tsv     agreement + the refitted Δ-logπ contrasts
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stats.inv_dir_recur_model import (FLOOR_QUANTILE, INVINFO_TSV,
                                           MIN_FLOOR, OUTPUT_CSV,
                                           choose_floor_from_quantile,
                                           load_and_match, run_model_A)
except Exception:  # pragma: no cover - loose-script fallback
    from inv_dir_recur_model import (FLOOR_QUANTILE, INVINFO_TSV,  # type: ignore
                                     MIN_FLOOR, OUTPUT_CSV,
                                     choose_floor_from_quantile,
                                     load_and_match, run_model_A)

SD_SIZE = "Flanking_inverted_repeat_size.kbp."
SD_IDENT = "Flanking_Inverted_repeat_identity"
CONSENSUS = "0_single_1_recur_consensus"

OUT_CALLS = "recurrence_sd_calls.tsv"
OUT_SUMMARY = "recurrence_sd_summary.tsv"


def _num(x):
    """Parse a cell that may carry a trailing '%' or be blank/NA."""
    if x is None:
        return np.nan
    s = str(x).strip().replace("%", "")
    if s in ("", "NA", "nan", "."):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_architecture(invinfo_tsv):
    """Per-locus SD size / identity and the consensus label, for classified loci."""
    inv = pd.read_csv(invinfo_tsv, sep="\t")
    inv = inv.rename(columns=lambda c: c.strip())
    df = pd.DataFrame({
        "chr_std": inv["Chromosome"].astype(str).str.replace("^chr", "", regex=True),
        "Start": pd.to_numeric(inv["Start"], errors="coerce"),
        "End": pd.to_numeric(inv["End"], errors="coerce"),
        "sd_size_kbp": inv[SD_SIZE].map(_num),
        "sd_identity_pct": inv[SD_IDENT].map(_num),
        "consensus": pd.to_numeric(inv[CONSENSUS], errors="coerce"),
    })
    df = df[df["consensus"].isin([0, 1])].copy()
    df = df.dropna(subset=["sd_size_kbp", "sd_identity_pct"])
    df["log10_sd_size"] = np.log10(df["sd_size_kbp"].clip(lower=1e-3))
    return df.reset_index(drop=True)


def _fit_logit(X, y):
    import statsmodels.api as sm
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)


def sd_calls(arch, threshold=0.5):
    """In-sample and leave-one-out SD-architecture recurrence calls."""
    import statsmodels.api as sm

    feats = ["log10_sd_size", "sd_identity_pct"]
    X = arch[feats].to_numpy(float)
    y = arch["consensus"].to_numpy(int)

    full = _fit_logit(X, y)
    p_in = full.predict(sm.add_constant(X, has_constant="add"))

    p_loo = np.empty(len(y))
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool)
        keep[i] = False
        if len(np.unique(y[keep])) < 2:
            p_loo[i] = np.nan
            continue
        fit = _fit_logit(X[keep], y[keep])
        p_loo[i] = float(fit.predict(
            sm.add_constant(X[i:i + 1], has_constant="add"))[0])

    out = arch.copy()
    out["p_recurrent_insample"] = p_in
    out["p_recurrent_loo"] = p_loo
    out["sd_call_insample"] = (p_in >= threshold).astype(int)
    out["sd_call_loo"] = np.where(np.isnan(p_loo), np.nan,
                                  (p_loo >= threshold).astype(float))
    return out, full, feats


def _agreement(out, col):
    ok = out[col].notna()
    agree = (out.loc[ok, col].astype(int) == out.loc[ok, "consensus"].astype(int))
    return float(agree.mean()), int(agree.sum()), int(ok.sum())


def refit_pi(matched, arch_calls, call_col, eps):
    """Refit the primary Delta-log pi model with SD-derived recurrence labels."""
    key = ["chr_std", "Start", "End"]
    lab = arch_calls[key + [call_col]].dropna(subset=[call_col]).copy()
    lab[call_col] = lab[call_col].astype(int)
    m = matched.copy()
    m["chr_std"] = m["chr_std"].astype(str)
    merged = m.merge(lab, on=key, how="inner")
    merged["Recurrence"] = np.where(merged[call_col] == 1, "Recurrent", "Single-event")
    _res, tab, _dfA = run_model_A(merged, eps)
    return tab, merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-csv", default=OUTPUT_CSV)
    ap.add_argument("--invinfo", default=INVINFO_TSV)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    print(">>> Non-circular recurrence from flanking SD architecture\n")
    arch = load_architecture(args.invinfo)
    print(f"Loci with a consensus call and complete SD architecture: {len(arch)} "
          f"({int((arch['consensus'] == 1).sum())} recurrent, "
          f"{int((arch['consensus'] == 0).sum())} single-event)")

    out, full, feats = sd_calls(arch, args.threshold)
    print("\nArchitecture-only logistic (no diversity term):")
    for name, coef, p in zip(["intercept"] + feats, full.params, full.pvalues):
        print(f"  {name:18s} beta = {coef:+8.4f}   p = {p:.4g}")

    acc_in, n_in, tot_in = _agreement(out, "sd_call_insample")
    acc_loo, n_loo, tot_loo = _agreement(out, "sd_call_loo")
    print(f"\nAgreement with the consensus calls:")
    print(f"  in-sample        {acc_in * 100:.1f}%  ({n_in}/{tot_in})")
    print(f"  leave-one-out    {acc_loo * 100:.1f}%  ({n_loo}/{tot_loo})")

    matched = load_and_match(args.output_csv, args.invinfo)
    # Same quantile-derived detection floor as the primary run, so the refitted
    # contrasts are comparable to the consensus-labelled ones line for line.
    all_pi = np.r_[matched["pi_direct"].to_numpy(float),
                   matched["pi_inverted"].to_numpy(float)]
    eps = choose_floor_from_quantile(all_pi, q=FLOOR_QUANTILE, min_floor=MIN_FLOOR)
    print(f"\nDetection floor eps = {eps:.6e}")

    rows = []
    rows.append({"quantity": "loci with consensus + SD architecture",
                 "value": len(arch), "p": ""})
    rows.append({"quantity": "SD-call agreement with consensus (in-sample)",
                 "value": f"{acc_in:.4f}", "p": ""})
    rows.append({"quantity": "SD-call agreement with consensus (leave-one-out)",
                 "value": f"{acc_loo:.4f}", "p": ""})

    for label, col in (("consensus", None),
                       ("SD architecture (in-sample)", "sd_call_insample"),
                       ("SD architecture (leave-one-out)", "sd_call_loo")):
        if col is None:
            _res, tab, dfA = run_model_A(matched, eps)
            n = len(dfA)
        else:
            tab, merged = refit_pi(matched, out, col, eps)
            n = len(merged)
        print(f"\n--- Delta-log pi, recurrence labelled by: {label}  (n = {n}) ---")
        for _, r in tab.iterrows():
            eff = r["ratio"]
            print(f"  {str(r['effect']):46s} fold-change = {float(eff):.3f}  "
                  f"p = {float(r['p']):.4g}")
            rows.append({"quantity": f"[{label}] {r['effect']}",
                         "value": f"{float(eff):.6f}", "p": f"{float(r['p']):.6g}"})

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, sep="\t", index=False)
    keep = ["chr_std", "Start", "End", "sd_size_kbp", "sd_identity_pct",
            "consensus", "p_recurrent_insample", "p_recurrent_loo",
            "sd_call_insample", "sd_call_loo"]
    out[keep].to_csv(OUT_CALLS, sep="\t", index=False)
    print(f"\nWrote {OUT_CALLS} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
