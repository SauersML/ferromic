#!/usr/bin/env python3
"""Run our trained PLS imputation models on a public cohort (1000 Genomes).

This is a thin, AoU-free wrapper around the inference logic in
``imputation/infer_dosage.py``. The production inference fills missing
genotypes using All-of-Us ancestry-specific means, which we deliberately do
NOT use here (no AoU resources for public benchmarking). Instead we fill any
missing genotype with the per-SNP global mean computed on the cohort itself,
which is the standard mean-imputation fallback and is adequate because the
1000 Genomes high-coverage panel is essentially complete at our tag SNPs.

Inputs (produced by imputation/prepare_data_for_infer.py):
  genotype_matrices/<model>.genotypes.npy   int8, shape (n_samples, n_snps)
  subset.fam                                PLINK fam (sample order)
  data/models/<model>.model.joblib          trained PLSRegression

Output:
  <out_tsv>  columns: SampleID, <model_id>  (imputed inverted-allele dosage)
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib

MISSING_VALUE_CODE = -127

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
GENOTYPE_DIR = os.getenv("GENOTYPE_DIR", "genotype_matrices")
PLINK_PREFIX = os.getenv("PLINK_PREFIX", "subset")


def global_mean_fill(X: np.ndarray) -> np.ndarray:
    """Replace MISSING_VALUE_CODE with the per-column (per-SNP) mean."""
    Xf = X.astype(np.float32, copy=True)
    missing = Xf == MISSING_VALUE_CODE
    if missing.any():
        col_means = np.zeros(Xf.shape[1], dtype=np.float32)
        for j in range(Xf.shape[1]):
            valid = Xf[:, j][~missing[:, j]]
            col_means[j] = valid.mean() if valid.size else 0.0
        idx = np.where(missing)
        Xf[idx] = col_means[idx[1]]
    return Xf


def main():
    if len(sys.argv) < 3:
        sys.exit("Usage: infer_dosage_public.py <model_id> <out_tsv>")
    model_id = sys.argv[1]
    out_tsv = sys.argv[2]

    fam = pd.read_csv(f"{PLINK_PREFIX}.fam", sep=r"\s+", header=None,
                      usecols=[1], dtype=str)
    sample_ids = fam[1].tolist()
    n = len(sample_ids)
    print(f"Samples: {n}")

    mat_path = os.path.join(GENOTYPE_DIR, f"{model_id}.genotypes.npy")
    X = np.load(mat_path, mmap_mode="r")
    X = np.array(X, copy=True)
    if X.shape[0] != n:
        sys.exit(f"Sample mismatch: matrix {X.shape[0]} vs fam {n}")

    miss_frac = float((X == MISSING_VALUE_CODE).mean())
    print(f"Model {model_id}: matrix {X.shape}, global missingness {miss_frac:.4%}")

    Xf = global_mean_fill(X)

    clf = joblib.load(os.path.join(MODEL_DIR, f"{model_id}.model.joblib"))
    pred = np.asarray(clf.predict(Xf)).reshape(-1).astype(np.float32)
    # Dosage is biologically in [0, 2]; clip to that range for reporting.
    pred = np.clip(pred, 0.0, 2.0)

    out = pd.DataFrame({"SampleID": sample_ids, model_id: pred})
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out)} dosages to {out_tsv}")
    print(out[model_id].describe())


if __name__ == "__main__":
    main()
