#!/usr/bin/env python3

import pandas as pd
import numpy as np
import subprocess

# Still need to handle ancestry
# Still need to loop over disease / inv pairs

# Existing PLINK data
ARRAYS_PREFIX = "arrays"              # arrays.bed / arrays.bim / arrays.fam

# Inversion hard calls (already discretized to 0/1/2)
INV_CALLS_TSV = "inv_hard_calls.tsv"  # FID IID inv_genotype

# Covariates input: must have FID, IID, sex, age, and PC columns (PC1, PC2, ...)
COVARIATES_TSV = "covariates_input.tsv"

# Phenotype input: must have FID, IID, and a binary column disease_name (0/1)
PHENOTYPE_TSV = "phenotype_input.tsv"
DISEASE_COL_NAME = "disease_name"

# Inversion pseudo-SNP definition
INV_SNP_ID = "INV_17Q21"
INV_CHR = "chr17"
INV_BP = 45585160  # set to a sensible coordinate for the inversion

# Intermediate and output prefixes
KEEP_SAMPLES_FILE = "bolt_keep_samples.txt"
INV_ONLY_PREFIX = "inv_only"
ARRAYS_KEEP_PREFIX = "arrays_keep"
ARRAYS_PLUS_INV_PREFIX = "arrays_plus_inv"

BOLT_COV_PATH = "bolt.cov"
BOLT_PHENO_PATH = "bolt.pheno"
BOLT_MODEL_SNPS_PATH = "bolt.modelSnps"


def load_fam(path):
    cols = ["FID", "IID", "father", "mother", "sex", "pheno"]
    fam = pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype=str)
    return fam


def load_bim(path):
    cols = ["CHR", "SNP", "GENPOS", "BP", "A1", "A0"]
    bim = pd.read_csv(path, sep=r"\s+", header=None, names=cols, dtype=str)
    return bim


def load_inv_calls(path):
    inv = pd.read_csv(path, sep="\t", dtype=str)
    inv = inv.rename(columns={"fid": "FID", "iid": "IID"})
    if "FID" not in inv.columns or "IID" not in inv.columns:
        raise ValueError("inv_hard_calls.tsv must have FID and IID columns.")
    if "inv_genotype" not in inv.columns:
        raise ValueError("inv_hard_calls.tsv must have an inv_genotype column.")
    inv["inv_genotype"] = pd.to_numeric(inv["inv_genotype"], errors="coerce")
    return inv[["FID", "IID", "inv_genotype"]]


def determine_keep_samples():
    fam = load_fam(f"{ARRAYS_PREFIX}.fam")
    inv = load_inv_calls(INV_CALLS_TSV)

    merged = fam.merge(inv, on=["FID", "IID"], how="left")
    missing_mask = merged["inv_genotype"].isna()

    n_total = len(fam)
    n_missing = int(missing_mask.sum())
    n_keep = n_total - n_missing

    print(f"Total individuals in arrays.fam: {n_total}")
    print(f"Individuals missing inversion hard calls (will be skipped): {n_missing}")
    print(f"Individuals kept with valid inversion genotype: {n_keep}")

    fam_keep = merged.loc[~missing_mask, ["FID", "IID", "father", "mother", "sex", "pheno"]].copy()
    inv_keep = merged.loc[~missing_mask, ["FID", "IID", "inv_genotype"]].copy()

    fam_keep.to_csv(KEEP_SAMPLES_FILE, sep="\t", header=False, index=False)

    return fam_keep, inv_keep


def prepare_inv_only_ped_and_map(fam_keep, inv_keep):
    combined = fam_keep.merge(inv_keep, on=["FID", "IID"], how="inner")
    if combined["inv_genotype"].isna().any():
        raise ValueError("Unexpected NaNs in inv_genotype after filtering.")

    def genotype_to_alleles(g):
        if g == 0:
            return ("A", "A")
        if g == 1:
            return ("A", "G")
        if g == 2:
            return ("G", "G")
        return ("0", "0")

    alleles = combined["inv_genotype"].apply(genotype_to_alleles)
    alleles = np.vstack(alleles.to_numpy())

    ped = combined[["FID", "IID", "father", "mother", "sex", "pheno"]].copy()
    ped = ped.fillna("0")
    ped = ped.astype(str)

    ped_geno = pd.DataFrame(alleles, columns=[f"{INV_SNP_ID}_A1", f"{INV_SNP_ID}_A2"])
    ped_full = pd.concat([ped, ped_geno], axis=1)
    ped_full.to_csv(f"{INV_ONLY_PREFIX}.ped", sep="\t", header=False, index=False)

    map_df = pd.DataFrame(
        [[INV_CHR, INV_SNP_ID, "0", str(INV_BP)]],
        columns=["CHR", "SNP", "GENPOS", "BP"],
    )
    map_df.to_csv(f"{INV_ONLY_PREFIX}.map", sep="\t", header=False, index=False)


def make_inv_only_bed():
    cmd = [
        "plink",
        "--file", INV_ONLY_PREFIX,
        "--make-bed",
        "--out", INV_ONLY_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def make_arrays_keep():
    cmd = [
        "plink",
        "--bfile", ARRAYS_PREFIX,
        "--keep", KEEP_SAMPLES_FILE,
        "--make-bed",
        "--out", ARRAYS_KEEP_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def merge_arrays_with_inv():
    cmd = [
        "plink",
        "--bfile", ARRAYS_KEEP_PREFIX,
        "--bmerge", f"{INV_ONLY_PREFIX}.bed", f"{INV_ONLY_PREFIX}.bim", f"{INV_ONLY_PREFIX}.fam",
        "--make-bed",
        "--out", ARRAYS_PLUS_INV_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def write_bolt_cov():
    fam = load_fam(f"{ARRAYS_PLUS_INV_PREFIX}.fam")
    cov = pd.read_csv(COVARIATES_TSV, sep="\t", dtype=str)
    cov = cov.rename(columns={"fid": "FID", "iid": "IID"})

    required_cols = {"FID", "IID", "sex", "age"}
    missing_required = required_cols - set(cov.columns)
    if missing_required:
        raise ValueError(f"Covariates file is missing required columns: {missing_required}")

    pc_cols = [c for c in cov.columns if c.startswith("PC")]
    cov["age"] = pd.to_numeric(cov["age"], errors="coerce")
    cov["age2"] = cov["age"] ** 2

    cov_use = cov[["FID", "IID", "sex", "age", "age2"] + pc_cols].copy()

    merged = fam[["FID", "IID"]].merge(cov_use, on=["FID", "IID"], how="left")
    if merged.isna().any(axis=None):
        bad = merged[merged.isna().any(axis=1)][["FID", "IID"]]
        raise ValueError(
            f"Missing covariates for some samples (no imputation performed). "
            f"Example rows:\n{bad.head()}"
        )

    merged.to_csv(BOLT_COV_PATH, sep="\t", header=True, index=False)
    print(f"Wrote covariate file: {BOLT_COV_PATH}")


def write_bolt_pheno():
    fam = load_fam(f"{ARRAYS_PLUS_INV_PREFIX}.fam")
    pheno = pd.read_csv(PHENOTYPE_TSV, sep="\t", dtype=str)
    pheno = pheno.rename(columns={"fid": "FID", "iid": "IID"})

    if DISEASE_COL_NAME not in pheno.columns:
        raise ValueError(f"{DISEASE_COL_NAME} not found in {PHENOTYPE_TSV}")

    pheno_use = pheno[["FID", "IID", DISEASE_COL_NAME]].copy()

    merged = fam[["FID", "IID"]].merge(pheno_use, on=["FID", "IID"], how="left")
    if merged[DISEASE_COL_NAME].isna().any():
        bad = merged[merged[DISEASE_COL_NAME].isna()][["FID", "IID"]]
        raise ValueError(
            f"Missing phenotype for some samples (no imputation performed). "
            f"Example rows:\n{bad.head()}"
        )

    vals = pd.to_numeric(merged[DISEASE_COL_NAME], errors="coerce")
    if not set(vals.unique()) <= {0, 1}:
        raise ValueError("Phenotype column must be strictly binary 0/1 with no other values.")

    merged.to_csv(BOLT_PHENO_PATH, sep="\t", header=True, index=False)
    print(f"Wrote phenotype file: {BOLT_PHENO_PATH}")


def write_model_snps():
    bim = load_bim(f"{ARRAYS_PLUS_INV_PREFIX}.bim")
    rows = []
    for snp in bim["SNP"]:
        if snp == INV_SNP_ID:
            rows.append((snp, "inv_component"))
        else:
            rows.append((snp, "background"))
    out = pd.DataFrame(rows, columns=["SNP_ID", "component_name"])
    out.to_csv(BOLT_MODEL_SNPS_PATH, sep="\t", header=False, index=False)
    print(f"Wrote model SNPs file: {BOLT_MODEL_SNPS_PATH}")


def main():
    fam_keep, inv_keep = determine_keep_samples()
    prepare_inv_only_ped_and_map(fam_keep, inv_keep)
    make_inv_only_bed()
    make_arrays_keep()
    merge_arrays_with_inv()
    write_bolt_cov()
    write_bolt_pheno()
    write_model_snps()


if __name__ == "__main__":
    main()
