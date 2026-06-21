#!/usr/bin/env python3
"""Compare our imputed inversion dosages against scoreInvHap genotypes.

For each inversion we join the two per-sample tables on SampleID and compute:
  - Pearson r and r^2 between our continuous dosage and the scoreInvHap dosage
  - Spearman rho (rank concordance, robust to the discrete scoreInvHap calls)
  - hard-call concordance: fraction of samples where round(our dosage) equals
    the scoreInvHap integer genotype
  - allele-frequency of the inverted allele under each method

Emits a tidy comparison TSV and a scatter PDF (one panel per inversion).

Usage:
  compute_concordance.py --out-tsv data/scoreinvhap_concordance.tsv \
      --out-pdf data/scoreinvhap_concordance.pdf \
      --pair NAME OUR_TSV OUR_COL SIH_TSV [--pair ...]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pair(name, our_tsv, our_col, sih_tsv):
    our = pd.read_csv(our_tsv, sep="\t", dtype={"SampleID": str})
    sih = pd.read_csv(sih_tsv, sep="\t", dtype={"SampleID": str})
    if our_col not in our.columns:
        # tolerate the model id being the only non-SampleID column
        cols = [c for c in our.columns if c != "SampleID"]
        if len(cols) == 1:
            our_col = cols[0]
        else:
            sys.exit(f"[{name}] column {our_col} not in {our.columns.tolist()}")
    our = our[["SampleID", our_col]].rename(columns={our_col: "our_dosage"})
    sih = sih[["SampleID", "scoreInvHap_dosage", "scoreInvHap_class"]]
    m = our.merge(sih, on="SampleID", how="inner")
    m = m.dropna(subset=["our_dosage", "scoreInvHap_dosage"])
    m["name"] = name
    m["model_col"] = our_col
    return m


def stats_for(name, m):
    x = m["our_dosage"].to_numpy(dtype=float)
    y = m["scoreInvHap_dosage"].to_numpy(dtype=float)
    n = len(m)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        r = rho = np.nan
    else:
        r = pearsonr(x, y)[0]
        rho = spearmanr(x, y)[0]
    hard = float(np.mean(np.round(np.clip(x, 0, 2)).astype(int) ==
                         np.round(y).astype(int)))
    return {
        "inversion": name,
        "model_id": m["model_col"].iloc[0],
        "n_samples": n,
        "pearson_r": round(r, 4) if r == r else np.nan,
        "r2": round(r * r, 4) if r == r else np.nan,
        "spearman_rho": round(rho, 4) if rho == rho else np.nan,
        "hardcall_concordance": round(hard, 4),
        "our_inv_allele_freq": round(float(np.mean(x) / 2.0), 4),
        "sih_inv_allele_freq": round(float(np.mean(y) / 2.0), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--pair", nargs=4, action="append", metavar=
                    ("NAME", "OUR_TSV", "OUR_COL", "SIH_TSV"), required=True)
    args = ap.parse_args()

    merged = [load_pair(*p) for p in args.pair]
    rows = [stats_for(m["name"].iloc[0], m) for m in merged]
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_tsv, sep="\t", index=False)
    print(summary.to_string(index=False))

    npan = len(merged)
    fig, axes = plt.subplots(1, npan, figsize=(5.2 * npan, 5), squeeze=False)
    for ax, m, row in zip(axes[0], merged, rows):
        jitter = (np.random.RandomState(0).rand(len(m)) - 0.5) * 0.12
        ax.scatter(m["scoreInvHap_dosage"] + jitter, m["our_dosage"],
                   s=8, alpha=0.35, edgecolors="none")
        ax.plot([0, 2], [0, 2], color="red", lw=1, ls="--")
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(-0.3, 2.3)
        ax.set_xlabel("scoreInvHap genotype (jittered)")
        ax.set_ylabel("Our imputed dosage")
        r2 = row["r2"]
        ax.set_title(f"{row['inversion']}\n"
                     f"r2={r2}  n={row['n_samples']}  "
                     f"hard={row['hardcall_concordance']}")
    fig.tight_layout()
    fig.savefig(args.out_pdf)
    print(f"Wrote {args.out_tsv} and {args.out_pdf}")


if __name__ == "__main__":
    main()
