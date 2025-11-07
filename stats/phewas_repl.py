import os
import sys
import pandas as pd
import polars as pl
import hail as hl

from PheTK.Cohort import Cohort
from PheTK.Phecode import Phecode
from PheTK.PheWAS import PheWAS
import PheTK


FERROMIC_URL = (
    "https://raw.githubusercontent.com/"
    "SauersML/ferromic/refs/heads/main/data/phewas_results.tsv"
)


# ======================= label normalization & ferromic panel =======================

def norm_label(s: str) -> str:
    """
    Normalize phenotype strings:
      'Melanocytic nevi' -> 'melanocytic_nevi'
      'Melanocytic_nevi' -> 'melanocytic_nevi'
    """
    s = s.strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    return "".join(out).strip("_")


def get_target_phecodes_from_ferromic():
    """
    Build the target phecodeX panel from ferromic:

    1. Load ferromic PheWAS results.
    2. Take unique 'Phenotype' labels.
    3. Normalize.
    4. Map to PheTK phecodeX via normalized phecode_string.
    5. Return sorted unique phecode list.
    """
    print(f"Loading ferromic PheWAS results from: {FERROMIC_URL}")
    ferro = pd.read_csv(FERROMIC_URL, sep="\t", low_memory=False)

    if "Phenotype" not in ferro.columns:
        raise RuntimeError(
            "Expected 'Phenotype' column not found in ferromic TSV. "
            f"Columns present: {list(ferro.columns)[:20]}"
        )

    labels = (
        ferro["Phenotype"]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    labels = sorted({lab for lab in labels if lab})
    print(f"Found {len(labels)} unique Phenotype labels in ferromic file.")

    # Load PheTK phecodeX mapping from installed package
    phetk_dir = os.path.dirname(PheTK.__file__)
    phecode_map_path = os.path.join(phetk_dir, "phecode", "phecodeX.csv")
    if not os.path.exists(phecode_map_path):
        raise RuntimeError(f"phecodeX.csv not found at {phecode_map_path}")

    phe_map = pl.read_csv(
        phecode_map_path,
        schema_overrides={
            "phecode": pl.Utf8,
            "ICD": pl.Utf8,
            "flag": pl.Int8,
            "code_val": pl.Float64,
            "phecode_string": pl.Utf8,
        },
        infer_schema_length=10000,
    )

    if "phecode_string" not in phe_map.columns:
        raise RuntimeError(
            "phecodeX.csv does not contain 'phecode_string' column; "
            "cannot map ferromic Phenotype labels."
        )

    # norm(phecode_string) -> phecode
    mapping = {}
    for row in phe_map.iter_rows(named=True):
        ps = row["phecode_string"]
        phe = row["phecode"]
        if ps is None or phe is None:
            continue
        key = norm_label(str(ps))
        if key and key not in mapping:
            mapping[key] = str(phe)

    targets = set()
    unmapped = []

    for lab in labels:
        key = norm_label(lab)
        phe = mapping.get(key)
        if phe is not None:
            targets.add(phe)
        else:
            unmapped.append(lab)

    print(
        f"Mapped {len(targets)} / {len(labels)} ferromic Phenotype labels "
        "to PheTK phecodeX codes via normalized phecode_string."
    )

    if not targets:
        example_unmapped = unmapped[:20]
        raise RuntimeError(
            "No ferromic Phenotype values mapped to PheTK phecodeX codes. "
            f"Example unmapped labels: {example_unmapped}"
        )

    targets = sorted(targets)
    print(f"Using {len(targets)} unique phecodeX codes as the phenotype panel.")
    return targets


# ======================= rs1052553 cohort: local or microarray MT ====================

def build_cohort_from_local_genotype():
    """
    If RS1052553_GENOTYPE_CSV is set, use that file.

    Expected columns:
      - person_id
      - rs1052553_dosage  (or rs1052553 / dosage)

    Returns path to normalized CSV or None if env var not set.
    """
    path = os.getenv("RS1052553_GENOTYPE_CSV")
    if not path:
        return None
    if not os.path.exists(path):
        raise RuntimeError(
            f"RS1052553_GENOTYPE_CSV is set to '{path}' but file does not exist."
        )

    print(f"Using precomputed genotype cohort from: {path}")
    df = pl.read_csv(path, infer_schema_length=1000)
    cols = set(df.columns)

    if "person_id" not in cols:
        raise RuntimeError(
            f"Genotype file {path} must contain 'person_id' column. "
            f"Columns: {df.columns}"
        )

    dosage_col = None
    for cand in ("rs1052553_dosage", "rs1052553", "dosage"):
        if cand in cols:
            dosage_col = cand
            break

    if dosage_col is None:
        raise RuntimeError(
            f"Genotype file {path} must contain rs1052553 dosage column "
            f"(one of rs1052553_dosage, rs1052553, dosage). "
            f"Columns: {df.columns}"
        )

    df = df.select(["person_id", dosage_col])
    df = df.rename({dosage_col: "rs1052553_dosage"})
    df = df.drop_nulls(["rs1052553_dosage"])
    df = df.with_columns(
        pl.col("person_id").cast(pl.Int64),
        pl.col("rs1052553_dosage").cast(pl.Int64),
    )
    df = df.unique(subset=["person_id"])

    out = "cohort_rs1052553_additive_raw.csv"
    df.write_csv(out)
    print(f"Normalized local genotype cohort written to {out} (n={df.height}).")
    return out


def build_cohort_from_microarray_hail():
    """
    Fallback: derive rs1052553 dosage from AoU microarray Hail MT.

    Uses:
      MICROARRAY_HAIL_STORAGE_PATH

    Steps:
      - Filter rows by rsid/ID == 'rs1052553'.
      - Compute GT.n_alt_alleles() as rs1052553_dosage.
      - Export via hl.export_table (no to_pandas on full cohort).
      - Normalize to CSV: person_id, rs1052553_dosage.

    Returns:
      path to CSV, or None if MICROARRAY_HAIL_STORAGE_PATH not set.
    """
    micro_mt_path = os.getenv("MICROARRAY_HAIL_STORAGE_PATH")
    if not micro_mt_path:
        return None

    print("Initializing Hail to derive rs1052553 cohort from microarray MT...")
    hl.init(app_name="rs1052553_additive_from_microarray",
            quiet=True,
            log="/tmp/hail_rs1052553_microarray.log")

    print(f"Reading microarray MT: {micro_mt_path}")
    mt = hl.read_matrix_table(micro_mt_path)

    row_fields = set(mt.row)
    if "rsid" in row_fields:
        mt_snp = mt.filter_rows(mt.rsid == "rs1052553")
    elif "ID" in row_fields:
        mt_snp = mt.filter_rows(mt.ID == "rs1052553")
    else:
        hl.stop()
        raise RuntimeError(
            "Microarray MT does not have 'rsid' or 'ID' row field; "
            "cannot locate rs1052553."
        )

    n = mt_snp.count_rows()
    if n == 0:
        hl.stop()
        raise RuntimeError(
            "rs1052553 not found in MICROARRAY_HAIL_STORAGE_PATH by rsid/ID."
        )

    print(f"Found rs1052553 with {n} row(s) in microarray MT.")

    if "GT" not in mt_snp.entry:
        hl.stop()
        raise RuntimeError(
            "Microarray MT for rs1052553 lacks 'GT' entry field; "
            "cannot compute additive dosage."
        )

    # Additive dosage = # alt alleles = 0/1/2
    mt_snp = mt_snp.select_entries(
        rs1052553_dosage=mt_snp.GT.n_alt_alleles()
    )
    ht = mt_snp.entries().select("rs1052553_dosage")

    out_tsv = "cohort_rs1052553_additive_raw.tsv"
    print(f"Exporting rs1052553 dosages to {out_tsv} via Hail export_table...")
    hl.export_table(ht, out_tsv, header=True)
    hl.stop()

    # Now load the tiny table locally and normalize
    print("Reading exported TSV and normalizing to CSV...")
    df = pl.read_csv(out_tsv, sep="\t", infer_schema_length=1000)

    # Identify sample column (whatever is not locus/alleles/dosage)
    candidate_sample_cols = [
        c for c in df.columns
        if c not in ("locus", "alleles", "rs1052553_dosage")
    ]
    if not candidate_sample_cols:
        raise RuntimeError(
            f"Could not identify sample ID column in exported table. "
            f"Columns: {df.columns}"
        )
    if len(candidate_sample_cols) > 1:
        print(
            f"Warning: multiple candidate sample ID columns found "
            f"{candidate_sample_cols}; using {candidate_sample_cols[0]}"
        )

    sample_col = candidate_sample_cols[0]

    df = df.select([sample_col, "rs1052553_dosage"])
    df = df.rename({sample_col: "person_id"})
    df = df.drop_nulls(["rs1052553_dosage"])
    df = df.with_columns(
        pl.col("person_id").cast(pl.Int64),
        pl.col("rs1052553_dosage").cast(pl.Int64),
    )
    df = df.unique(subset=["person_id"])

    out_csv = "cohort_rs1052553_additive_raw.csv"
    df.write_csv(out_csv)
    print(f"Additive rs1052553 cohort written to {out_csv} (n={df.height}).")
    return out_csv


def build_rs1052553_additive_cohort():
    """
    Orchestrate genotype cohort:

    1. If RS1052553_GENOTYPE_CSV is set -> normalize & use that.
    2. Else, try to derive from MICROARRAY_HAIL_STORAGE_PATH via Hail.
    """
    path = build_cohort_from_local_genotype()
    if path:
        return path

    path = build_cohort_from_microarray_hail()
    if path:
        return path

    raise RuntimeError(
        "Unable to build rs1052553 cohort: "
        "set RS1052553_GENOTYPE_CSV to a local genotype file or "
        "ensure MICROARRAY_HAIL_STORAGE_PATH is defined and contains rs1052553."
    )


# ======================= AoU covariates & phecodeX counts ======================

def add_aou_covariates(cohort_path):
    """
    Use PheTK Cohort(platform='aou') to add:
      - age_at_last_event
      - sex_at_birth
      - first 10 PCs
    """
    print(f"Adding AoU covariates to cohort: {cohort_path}")
    cohort = Cohort(platform="aou", aou_db_version=8)

    cohort.add_covariates(
        cohort_csv_path=cohort_path,
        age_at_last_event=True,
        sex_at_birth=True,
        first_n_pcs=10,
        drop_nulls=True,
        output_file_name="cohort_rs1052553_additive_cov.csv",
    )

    out = "cohort_rs1052553_additive_cov.csv"
    print(f"Cohort with covariates written to {out}")
    return out


def build_phecode_counts_x():
    """
    Use AoU EHR only to get phecodeX counts via PheTK.
    """
    print("Building phecodeX counts from All of Us EHR via PheTK...")
    phe = Phecode(platform="aou")

    phe.count_phecode(
        phecode_version="X",
        icd_version="US",
        phecode_map_file_path=None,
        output_file_name="phecode_counts_x_all.csv",
    )

    out = "phecode_counts_x_all.csv"
    print(f"All phecodeX counts written to {out}")
    return out


def subset_phecode_counts(phecode_counts_path, target_phecodes):
    """
    Restrict phecode counts to ferromic-derived phecodes.
    """
    print(
        f"Subsetting {phecode_counts_path} to "
        f"{len(target_phecodes)} target phecodes from ferromic panel..."
    )

    phe_all = pl.read_csv(phecode_counts_path, infer_schema_length=10000)

    if "phecode" not in phe_all.columns:
        raise RuntimeError(
            f"'phecode' column not found in {phecode_counts_path}."
        )

    phe_all = phe_all.with_columns(pl.col("phecode").cast(pl.Utf8))
    phe_sub = phe_all.filter(pl.col("phecode").is_in(target_phecodes))

    if phe_sub.height == 0:
        raise RuntimeError(
            "No rows in phecode counts matched the target phecodes. "
            "Check ferromic-to-phecodeX mapping."
        )

    out = "phecode_counts_x_ferromic_panel.csv"
    phe_sub.write_csv(out)

    present = sorted(set(phe_sub["phecode"].to_list()))
    print(
        f"Subset written to {out} with {len(present)} distinct phecodes "
        f"present out of {len(target_phecodes)} requested."
    )

    return out


# ======================= covariate inference & PheWAS run ======================

def infer_covariates_from_cohort(cohort_path):
    """
    Inspect cohort file to choose:
      - sex_at_birth_col
      - covariate_cols (age + PCs)
    """
    df = pl.read_csv(cohort_path, n_rows=5, infer_schema_length=1000)
    cols = set(df.columns)

    if "sex_at_birth" in cols:
        sex_col = "sex_at_birth"
    elif "sex" in cols:
        sex_col = "sex"
    else:
        raise RuntimeError(
            f"No sex column ('sex_at_birth' or 'sex') found in {cohort_path}. "
            f"Columns: {df.columns}"
        )

    covariates = []

    if "age_at_last_event" in cols:
        covariates.append("age_at_last_event")
    if "natural_age" in cols:
        covariates.append("natural_age")

    for i in range(1, 21):
        pc = f"pc{i}"
        if pc in cols:
            covariates.append(pc)

    if not covariates:
        raise RuntimeError(
            "No covariate columns (age_at_last_event/natural_age or pc1+) "
            f"detected in cohort file. Columns: {df.columns}"
        )

    return sex_col, covariates


def run_phewas(cohort_path, phecode_counts_path, target_phecodes):
    """
    Run PheTK PheWAS:
      - phecode_version = X
      - independent_variable_of_interest = rs1052553_dosage
      - restricted to target_phecodes
    """
    print("Inferring covariates from enriched cohort...")
    sex_col, covariate_cols = infer_covariates_from_cohort(cohort_path)

    print(f"Using sex_at_birth_col: {sex_col}")
    print(f"Using covariates: {covariate_cols}")

    print(
        "Running PheWAS for rs1052553 (additive dosage) "
        f"on ferromic-derived phecode panel ({len(target_phecodes)} phecodes)..."
    )

    phewas = PheWAS(
        phecode_version="X",
        phecode_count_csv_path=phecode_counts_path,
        cohort_csv_path=cohort_path,
        sex_at_birth_col=sex_col,
        male_as_one=True,
        covariate_cols=covariate_cols,
        independent_variable_of_interest="rs1052553_dosage",
        min_cases=50,
        min_phecode_count=2,
        phecode_to_process=target_phecodes,
        output_file_name="phewas_rs1052553_additive_ferromic_panel.csv",
    )

    phewas.run()
    print(
        "PheWAS complete. Results written to "
        "phewas_rs1052553_additive_ferromic_panel.csv"
    )


# ======================= main ======================

def main():
    cdr = os.getenv("WORKSPACE_CDR")
    print("Workspace CDR:", cdr)

    if not cdr:
        print(
            "ERROR: WORKSPACE_CDR not set. "
            "Run this inside an All of Us Researcher Workbench environment."
        )
        sys.exit(1)

    # 1. Phenotype panel from ferromic
    target_phecodes = get_target_phecodes_from_ferromic()

    # 2. rs1052553 additive cohort:
    #    - from RS1052553_GENOTYPE_CSV if set
    #    - else from MICROARRAY_HAIL_STORAGE_PATH via Hail
    cohort_raw = build_rs1052553_additive_cohort()

    # 3. Add AoU covariates (AoU only)
    cohort_cov = add_aou_covariates(cohort_raw)

    # 4. Build phecodeX counts from AoU EHR (AoU only)
    phecode_counts_all = build_phecode_counts_x()

    # 5. Subset to ferromic-derived phecodes
    phecode_counts_subset = subset_phecode_counts(
        phecode_counts_all,
        target_phecodes,
    )

    # 6. Run PheWAS
    run_phewas(
        cohort_cov,
        phecode_counts_subset,
        target_phecodes,
    )

    print("Done.")


if __name__ == "__main__":
    main()
