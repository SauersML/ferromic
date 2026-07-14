"""Stage 3 (score): apply the sim-fit transferable classifier to the real inversions.

Builds the transferable pop-gen feature table for the real inversions from ferromic's
canonical per-inversion ``output.csv`` (matched to ``inv_properties.tsv``), scores each
usable inversion to a continuous recurrence score + binary call, and measures
concordance with the manuscript's consensus single/recurrent calls.

The binary-call threshold is chosen on the held-out simulation negatives so
FPR <= FMAX (the operating point the classifier optimizes), then applied to the real
scores. The concordance against the consensus labels is a consistency check, not
independent validation, because the consensus labels are themselves derived from
tag-SNP + parsimony signals that overlap several classifier features; the non-circular
validation is the simulation ground truth (see the ``fit`` stage). This module records
the number; it draws no conclusion from it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

from . import classifier as C
from . import paths
from .transferable import TRANSFERABLE_FEATURES, from_output_csv_row


def build_real_features(inv_properties=None, output_csv=None):
    """Transferable feature table for the real inversions from output.csv."""
    inv_properties = inv_properties or paths.INV_PROPERTIES
    output_csv = output_csv or paths.OUTPUT_CSV
    inv = pd.read_csv(inv_properties, sep="\t").dropna(subset=["Chromosome"])
    o = pd.read_csv(output_csv)
    o["chr"] = o["chr"].astype(str)
    o = o.set_index(["chr", "region_start"])
    rows = []
    for _, r in inv.iterrows():
        ch = str(r["Chromosome"]).replace("chr", "")
        key = (ch, int(r["Start"]))
        rec = dict(chrom=str(r["Chromosome"]), start=int(r["Start"]), end=int(r["End"]),
                   inv_id=r["OrigID"], consensus=r["0_single_1_recur_consensus"],
                   inv_AF=r.get("Inverted_AF"), size_kbp=r.get("Size_.kbp."))
        if key in o.index:
            orow = o.loc[key]
            if getattr(orow, "ndim", 1) > 1:
                orow = orow.iloc[0]
            nh0 = float(orow.get("0_num_hap_filter", 0) or 0)
            nh1 = float(orow.get("1_num_hap_filter", 0) or 0)
            rec.update(from_output_csv_row(orow))
            rec.update(dict(n_hap0=nh0, n_hap1=nh1, matched=True,
                            usable=bool(nh0 >= 2 and nh1 >= 2)))
        else:
            rec.update({k: np.nan for k in TRANSFERABLE_FEATURES})
            rec.update(dict(n_hap0=0, n_hap1=0, matched=False, usable=False))
        rows.append(rec)
    return pd.DataFrame(rows)


def choose_threshold(sim_pred_csv=None, fmax=C.FMAX):
    """Highest-TPR threshold on the transferable held-out sim predictions with FPR<=fmax."""
    sim_pred_csv = sim_pred_csv or paths.TF_SIM_TEST_PRED
    df = pd.read_csv(sim_pred_csv)
    fpr, tpr, thr = roc_curve(df["label"].values, df["score"].values)
    ok = fpr <= fmax
    return float(thr[ok][np.argmax(tpr[ok])])


def run(model_json=None, inv_properties=None, output_csv=None, sim_pred=None, outdir=None):
    model_json = model_json or paths.MODEL_TRANSFERABLE
    outdir = outdir or paths.RESULTS
    os.makedirs(outdir, exist_ok=True)
    with open(model_json) as fh:
        model = C.Model(**json.load(fh))

    real = build_real_features(inv_properties, output_csv)
    thr = choose_threshold(sim_pred)

    usable = real["usable"] == True  # noqa: E712
    real["recurrence_score"] = np.nan
    real.loc[usable, "recurrence_score"] = model.score(real.loc[usable, TRANSFERABLE_FEATURES].values)
    real["recurrence_call"] = (real["recurrence_score"] >= thr).astype("Int64")
    real.loc[~usable, "recurrence_call"] = pd.NA
    real["low_confidence"] = (~usable) | (real["n_hap0"] < 4) | (real["n_hap1"] < 4)
    real.to_csv(os.path.join(outdir, "real_scores.csv"), index=False)

    lab = real[real["consensus"].isin([0, 1]) & usable].copy()
    y = lab["consensus"].astype(int).values
    s = lab["recurrence_score"].values
    call = lab["recurrence_call"].astype(int).values
    conc = {
        "n_consensus_scored": int(len(lab)),
        "auc_vs_consensus": float(roc_auc_score(y, s)) if len(set(y)) > 1 else None,
        "kappa_call_vs_consensus": float(cohen_kappa_score(y, call)),
        "agreement": float((call == y).mean()),
        "threshold_fpr_fmax": thr,
        "circularity_caveat": ("Concordance is partly circular: the consensus labels are "
                               "themselves derived from tag-SNP + parsimony signals that overlap "
                               "several classifier features. This is a consistency check, not "
                               "independent validation. The non-circular validation is the "
                               "simulation ground truth (sim_metrics.json)."),
    }
    dis = lab[call != y][["chrom", "start", "end", "inv_id", "consensus",
                          "recurrence_score", "recurrence_call", "inv_AF", "size_kbp",
                          "n_hap0", "n_hap1"]]
    conc["n_disagreements"] = int(len(dis))
    dis.to_csv(os.path.join(outdir, "concordance_disagreements.csv"), index=False)
    conc["coverage"] = {
        "n_total": int(len(real)),
        "n_usable": int(usable.sum()),
        "n_scored_nonconsensus": int((usable & real["consensus"].isna()).sum()),
        "n_called_recurrent": int((real["recurrence_call"] == 1).sum()),
        "n_low_confidence": int(real["low_confidence"].sum()),
    }
    with open(os.path.join(outdir, "concordance.json"), "w") as fh:
        json.dump(conc, fh, indent=2)
    paths.write_provenance(os.path.join(outdir, "score_provenance.json"),
                           {"model": model_json,
                            "inv_properties": inv_properties or paths.INV_PROPERTIES,
                            "output_csv": output_csv or paths.OUTPUT_CSV,
                            "sim_pred": sim_pred or paths.TF_SIM_TEST_PRED})
    print(json.dumps(conc, indent=2))
    return real, conc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the real inversions with the transferable classifier.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--inv-properties", default=None)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--sim-pred", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)
    run(model_json=args.model, inv_properties=args.inv_properties, output_csv=args.output_csv,
        sim_pred=args.sim_pred, outdir=args.outdir)


if __name__ == "__main__":
    main()
