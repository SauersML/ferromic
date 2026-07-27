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
1. **Primary: a hard rule fixed a priori.** Recurrent-prone iff flanking-repeat
   identity >= 95% AND repeat size >= 10 kbp -- the NAHR substrate thresholds,
   chosen from the mechanism, *not* fitted to the 93 labels. Nothing about the
   consensus classification enters the call, so this is non-circular in the
   strong sense: no fitting, no cross-validation needed.
2. **Supplementary: a fitted logistic** on the same two features, with
   leave-one-out cross-validation. Reported only as a sensitivity check. It is
   weaker on both counts -- it agrees with the consensus less often *and* it is
   trained on the labels, which the hard rule never touches.
3. Refits the primary Δ-logπ model (``inv_dir_recur_model.run_model_A``, HC3 SEs,
   the same quantile-derived epsilon) under each label set.

The hard rule is the one to quote. An earlier version of this script used only the
fitted logistic and reported 74.2% agreement; that was the wrong classifier --
both less concordant and, being label-fitted, less able to answer the circularity
charge it exists to answer.

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

# A-priori NAHR thresholds. Recurrent inversions recur by non-allelic homologous
# recombination between flanking segmental duplications, which needs repeats that
# are both long enough and similar enough to misalign. These two numbers are set
# from that mechanism and are never tuned against the consensus labels.
NAHR_MIN_IDENTITY_PCT = 95.0
NAHR_MIN_SIZE_KBP = 10.0

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


def hard_rule(arch):
    """The a-priori NAHR call: identity >= 95% AND repeat >= 10 kbp."""
    return ((arch["sd_identity_pct"] >= NAHR_MIN_IDENTITY_PCT)
            & (arch["sd_size_kbp"] >= NAHR_MIN_SIZE_KBP)).astype(int)


def nahr_score(arch):
    """Threshold-free NAHR potential: how far into the recurrent-prone corner a
    locus sits. Monotone in both features, so it orders loci the same way the hard
    rule splits them, without depending on where the thresholds fall."""
    return (np.log10(arch["sd_size_kbp"].clip(lower=1e-3))
            + (arch["sd_identity_pct"] - NAHR_MIN_IDENTITY_PCT) / 10.0)


def cohens_kappa(a, b):
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    n = len(a)
    po = float((a == b).mean())
    pe = float(((a == 1).mean() * (b == 1).mean())
               + ((a == 0).mean() * (b == 0).mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


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
    out["sd_call_hard"] = hard_rule(arch)
    out["nahr_score"] = nahr_score(arch)
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

    print("\nPRIMARY -- a-priori NAHR hard rule "
          f"(identity >= {NAHR_MIN_IDENTITY_PCT:.0f}% AND size >= {NAHR_MIN_SIZE_KBP:.0f} kbp)")
    acc_hard, n_hard, tot_hard = _agreement(out, "sd_call_hard")
    k_hard = cohens_kappa(out["sd_call_hard"], out["consensus"])
    tp = int(((out.sd_call_hard == 1) & (out.consensus == 1)).sum())
    fp = int(((out.sd_call_hard == 1) & (out.consensus == 0)).sum())
    fn = int(((out.sd_call_hard == 0) & (out.consensus == 1)).sum())
    tn = int(((out.sd_call_hard == 0) & (out.consensus == 0)).sum())
    print(f"  agreement with consensus  {acc_hard * 100:.1f}%  ({n_hard}/{tot_hard})   "
          f"kappa = {k_hard:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}   reclassified = {fp + fn}")
    for lab, sel in (("recurrent", out.consensus == 1), ("single-event", out.consensus == 0)):
        g = out[sel]
        print(f"  consensus {lab:12s} median identity {g.sd_identity_pct.median():.1f}%  "
              f"median size {g.sd_size_kbp.median():.2f} kbp")

    print("\nSUPPLEMENTARY -- fitted logistic (label-trained; sensitivity only)")
    for name, coef, pv in zip(["intercept"] + feats, full.params, full.pvalues):
        print(f"  {name:18s} beta = {coef:+8.4f}   p = {pv:.4g}")
    acc_in, n_in, tot_in = _agreement(out, "sd_call_insample")
    acc_loo, n_loo, tot_loo = _agreement(out, "sd_call_loo")
    print(f"  agreement in-sample     {acc_in * 100:.1f}%  ({n_in}/{tot_in})")
    print(f"  agreement leave-one-out {acc_loo * 100:.1f}%  ({n_loo}/{tot_loo})")

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
    rows.append({"quantity": "hard-rule agreement with consensus",
                 "value": f"{acc_hard:.4f}", "p": ""})
    rows.append({"quantity": "hard-rule Cohen kappa", "value": f"{k_hard:.4f}", "p": ""})
    rows.append({"quantity": "hard-rule loci reclassified", "value": fp + fn, "p": ""})
    rows.append({"quantity": "logistic agreement (in-sample)",
                 "value": f"{acc_in:.4f}", "p": ""})
    rows.append({"quantity": "logistic agreement (leave-one-out)",
                 "value": f"{acc_loo:.4f}", "p": ""})

    for label, col in (("consensus", None),
                       ("SD hard rule (primary)", "sd_call_hard"),
                       ("SD logistic (in-sample)", "sd_call_insample"),
                       ("SD logistic (leave-one-out)", "sd_call_loo")):
        if col is None:
            _res, tab, dfA = run_model_A(matched, eps)
            n = len(dfA)
        else:
            tab, merged = refit_pi(matched, out, col, eps)
            n = len(merged)
        print(f"\n--- Delta-log pi, recurrence labelled by: {label}  (n = {n}) ---")
        for _, r in tab.iterrows():
            print(f"  {str(r['effect']):46s} fold-change = {float(r['ratio']):.3f}  "
                  f"p = {float(r['p']):.4g}")
            rows.append({"quantity": f"[{label}] {r['effect']}",
                         "value": f"{float(r['ratio']):.6f}", "p": f"{float(r['p']):.6g}"})

    # Threshold-free control: does the outcome track the continuous NAHR score?
    from scipy import stats as _st
    key = ["chr_std", "Start", "End"]
    m2 = matched.copy()
    m2["chr_std"] = m2["chr_std"].astype(str)
    g = m2.merge(out[key + ["nahr_score"]], on=key, how="inner")
    g["dlogpi"] = (np.log(g["pi_inverted"].to_numpy(float) + eps)
                   - np.log(g["pi_direct"].to_numpy(float) + eps))
    rho, pv = _st.spearmanr(g["nahr_score"], g["dlogpi"])
    print(f"\nContinuous NAHR-score gradient (threshold-free): "
          f"Spearman rho = {rho:.3f}, p = {pv:.4g}  (n = {len(g)})")
    rows.append({"quantity": "continuous NAHR score vs Delta-log pi (Spearman rho)",
                 "value": f"{rho:.6f}", "p": f"{pv:.6g}"})

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, sep="\t", index=False)
    keep = ["chr_std", "Start", "End", "sd_size_kbp", "sd_identity_pct",
            "nahr_score", "consensus", "sd_call_hard",
            "p_recurrent_insample", "p_recurrent_loo",
            "sd_call_insample", "sd_call_loo"]
    out[keep].to_csv(OUT_CALLS, sep="\t", index=False)
    print(f"\nWrote {OUT_CALLS} and {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
