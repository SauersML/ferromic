import os
import json
import pandas as pd
import numpy as np
import subprocess

# Still need to handle ancestry
# Still need to loop over disease / inv pairs

ARRAYS_PREFIX = "arrays"
INV_CALLS_TSV = "inv_hard_calls.tsv"
COVARIATES_TSV = "covariates_input.tsv"
PHENOTYPE_TSV = "phenotype_input.tsv"
DISEASE_COL_NAME = "disease_name"

INV_SNP_ID = "INV_17Q21"
INV_CHR = "chr17"
INV_BP = 45585160

BR_CACHE_DIR = "br_cache"
CACHE_META_PATH = os.path.join(BR_CACHE_DIR, "prep_cache_state.json")

CACHED_KEEP_SAMPLES_FILE = os.path.join(BR_CACHE_DIR, "bolt_keep_samples.txt")
CACHED_INV_ONLY_PREFIX = os.path.join(BR_CACHE_DIR, "inv_only")
CACHED_ARRAYS_KEEP_PREFIX = os.path.join(BR_CACHE_DIR, "arrays_keep")
CACHED_ARRAYS_PLUS_INV_PREFIX = os.path.join(BR_CACHE_DIR, "arrays_plus_inv")

CACHED_BOLT_COV_PATH = os.path.join(BR_CACHE_DIR, "bolt.cov")
CACHED_BOLT_PHENO_PATH = os.path.join(BR_CACHE_DIR, "bolt.pheno")
CACHED_BOLT_MODEL_SNPS_PATH = os.path.join(BR_CACHE_DIR, "bolt.modelSnps")

WORK_ARRAYS_PLUS_INV_PREFIX = "arrays_plus_inv"
WORK_BOLT_COV_PATH = "bolt.cov"
WORK_BOLT_PHENO_PATH = "bolt.pheno"
WORK_BOLT_MODEL_SNPS_PATH = "bolt.modelSnps"

PLINK_THREADS = max(1, (os.cpu_count() or 1) // 2)


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


def file_fingerprint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def load_cache_meta():
    os.makedirs(BR_CACHE_DIR, exist_ok=True)
    if not os.path.exists(CACHE_META_PATH):
        return {"version": 1}
    with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != 1:
        return {"version": 1}
    return data


def save_cache_meta(meta):
    meta["version"] = 1
    tmp_path = CACHE_META_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, sort_keys=True, indent=2)
    os.replace(tmp_path, CACHE_META_PATH)


def compute_geno_key():
    inputs = {
        "arrays_bed": file_fingerprint(f"{ARRAYS_PREFIX}.bed"),
        "arrays_bim": file_fingerprint(f"{ARRAYS_PREFIX}.bim"),
        "arrays_fam": file_fingerprint(f"{ARRAYS_PREFIX}.fam"),
        "inv_calls": file_fingerprint(INV_CALLS_TSV),
        "inv_snp_id": INV_SNP_ID,
        "inv_chr": INV_CHR,
        "inv_bp": INV_BP,
    }
    return json.dumps(inputs, sort_keys=True)


def compute_cov_key(geno_key):
    inputs = {
        "geno_key": geno_key,
        "covariates": file_fingerprint(COVARIATES_TSV),
    }
    return json.dumps(inputs, sort_keys=True)


def compute_pheno_key(geno_key):
    inputs = {
        "geno_key": geno_key,
        "phenotype": file_fingerprint(PHENOTYPE_TSV),
        "disease_col": DISEASE_COL_NAME,
    }
    return json.dumps(inputs, sort_keys=True)


def compute_modelsnps_key(geno_key):
    inputs = {"geno_key": geno_key}
    return json.dumps(inputs, sort_keys=True)


def geno_outputs_exist():
    paths = [
        f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.bed",
        f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.bim",
        f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.fam",
        CACHED_KEEP_SAMPLES_FILE,
    ]
    for p in paths:
        if not os.path.exists(p):
            return False
    return True


def cov_outputs_exist():
    return os.path.exists(CACHED_BOLT_COV_PATH)


def pheno_outputs_exist():
    return os.path.exists(CACHED_BOLT_PHENO_PATH)


def modelsnps_outputs_exist():
    return os.path.exists(CACHED_BOLT_MODEL_SNPS_PATH)


def ensure_symlink(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


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

    fam_keep.to_csv(CACHED_KEEP_SAMPLES_FILE, sep="\t", header=False, index=False)

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
    ped_full.to_csv(f"{CACHED_INV_ONLY_PREFIX}.ped", sep="\t", header=False, index=False)

    map_df = pd.DataFrame(
        [[INV_CHR, INV_SNP_ID, "0", str(INV_BP)]],
        columns=["CHR", "SNP", "GENPOS", "BP"],
    )
    map_df.to_csv(f"{CACHED_INV_ONLY_PREFIX}.map", sep="\t", header=False, index=False)


def make_inv_only_bed():
    cmd = [
        "plink",
        "--threads", str(PLINK_THREADS),
        "--file", CACHED_INV_ONLY_PREFIX,
        "--make-bed",
        "--out", CACHED_INV_ONLY_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def make_arrays_keep():
    cmd = [
        "plink",
        "--threads", str(PLINK_THREADS),
        "--bfile", ARRAYS_PREFIX,
        "--keep", CACHED_KEEP_SAMPLES_FILE,
        "--make-bed",
        "--out", CACHED_ARRAYS_KEEP_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def merge_arrays_with_inv():
    cmd = [
        "plink",
        "--threads", str(PLINK_THREADS),
        "--bfile", CACHED_ARRAYS_KEEP_PREFIX,
        "--bmerge",
        f"{CACHED_INV_ONLY_PREFIX}.bed",
        f"{CACHED_INV_ONLY_PREFIX}.bim",
        f"{CACHED_INV_ONLY_PREFIX}.fam",
        "--make-bed",
        "--out", CACHED_ARRAYS_PLUS_INV_PREFIX,
    ]
    subprocess.run(cmd, check=True)


def write_bolt_cov():
    fam = load_fam(f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.fam")
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

    merged.to_csv(CACHED_BOLT_COV_PATH, sep="\t", header=True, index=False)
    print(f"Wrote covariate file: {CACHED_BOLT_COV_PATH}")


def write_bolt_pheno():
    fam = load_fam(f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.fam")
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

    merged.to_csv(CACHED_BOLT_PHENO_PATH, sep="\t", header=True, index=False)
    print(f"Wrote phenotype file: {CACHED_BOLT_PHENO_PATH}")


def write_model_snps():
    bim = load_bim(f"{CACHED_ARRAYS_PLUS_INV_PREFIX}.bim")
    rows = []
    for snp in bim["SNP"]:
        if snp == INV_SNP_ID:
            rows.append((snp, "inv_component"))
        else:
            rows.append((snp, "background"))
    out = pd.DataFrame(rows, columns=["SNP_ID", "component_name"])
    out.to_csv(CACHED_BOLT_MODEL_SNPS_PATH, sep="\t", header=False, index=False)
    print(f"Wrote model SNPs file: {CACHED_BOLT_MODEL_SNPS_PATH}")


def materialize_geno_outputs():
    for ext in (".bed", ".bim", ".fam"):
        src = f"{CACHED_ARRAYS_PLUS_INV_PREFIX}{ext}"
        dst = f"{WORK_ARRAYS_PLUS_INV_PREFIX}{ext}"
        ensure_symlink(src, dst)


def main():
    meta = load_cache_meta()

    geno_key = compute_geno_key()
    geno_cached = meta.get("geno_key") == geno_key and geno_outputs_exist()

    if not geno_cached:
        fam_keep, inv_keep = determine_keep_samples()
        prepare_inv_only_ped_and_map(fam_keep, inv_keep)
        make_inv_only_bed()
        make_arrays_keep()
        merge_arrays_with_inv()
        meta["geno_key"] = geno_key
        save_cache_meta(meta)
    else:
        print("Using cached genotype layer from br_cache.")
    materialize_geno_outputs()

    cov_key = compute_cov_key(geno_key)
    cov_cached = meta.get("cov_key") == cov_key and cov_outputs_exist()
    if not cov_cached:
        write_bolt_cov()
        meta["cov_key"] = cov_key
        save_cache_meta(meta)
    else:
        print("Using cached covariate file from br_cache.")
    ensure_symlink(CACHED_BOLT_COV_PATH, WORK_BOLT_COV_PATH)

    pheno_key = compute_pheno_key(geno_key)
    pheno_cached = meta.get("pheno_key") == pheno_key and pheno_outputs_exist()
    if not pheno_cached:
        write_bolt_pheno()
        meta["pheno_key"] = pheno_key
        save_cache_meta(meta)
    else:
        print("Using cached phenotype file from br_cache.")
    ensure_symlink(CACHED_BOLT_PHENO_PATH, WORK_BOLT_PHENO_PATH)

    modelsnps_key = compute_modelsnps_key(geno_key)
    modelsnps_cached = meta.get("modelsnps_key") == modelsnps_key and modelsnps_outputs_exist()
    if not modelsnps_cached:
        write_model_snps()
        meta["modelsnps_key"] = modelsnps_key
        save_cache_meta(meta)
    else:
        print("Using cached modelSnps file from br_cache.")
    ensure_symlink(CACHED_BOLT_MODEL_SNPS_PATH, WORK_BOLT_MODEL_SNPS_PATH)


if __name__ == "__main__":
    main()
