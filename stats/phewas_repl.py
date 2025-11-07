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


# -------------- helpers: label normalization & ferromic panel -----------------


def norm_label(s: str) -> str:
    """
    Normalize phenotype strings so label styles match between
    ferromic 'Phenotype' and PheTK 'phecode_string'.

    Examples:
      'Melanocytic_nevi' -> 'melanocytic_nevi'
      'Melanocytic nevi' -> 'melanocytic_nevi'
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
    1. Load ferromic PheWAS results from FERROMIC_URL.
    2. Use the 'Phenotype' column.
    3. Deduplicate.
    4. Map those phenotype labels to PheTK phecodeX definitions:
       - Normalize both ferromic Phenotype and phecodeX phecode_string.
       - Build mapping norm(phecode_string) -> phecode.
       - Keep only successfully mapped phecodes.
    5. Return a sorted list of unique phecodes.
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
        dtypes={
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

    # Build norm(phecode_string) -> phecode mapping
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


# -------------- helpers: rs1052553 cohort from AoU hail MTs -------------------


def select_mt_with_rs1052553():
    """
    Search for rs1052553 in available AoU v8 Hail MatrixTables by rsID.

    Tries, in order:
      - Exome split MT (WGS_EXOME_SPLIT_HAIL_PATH)
      - Exome multi MT (WGS_EXOME_MULTI_HAIL_PATH)
      - Microarray MT (MICROARRAY_HAIL_STORAGE_PATH)

    Returns:
      (label, mt_snp)
    where mt_snp is a MatrixTable filtered to rs1052553 rows.

    Raises:
      RuntimeError if rs1052553 not found in any candidate.
    """
    candidates = []

    exome_split = os.getenv("WGS_EXOME_SPLIT_HAIL_PATH")
    exome_multi = os.getenv("WGS_EXOME_MULTI_HAIL_PATH")
    micro_mt = os.getenv("MICROARRAY_HAIL_STORAGE_PATH")

    if exome_split:
        candidates.append(("WGS_EXOME_SPLIT_HAIL_PATH", exome_split))
    if exome_multi:
        candidates.append(("WGS_EXOME_MULTI_HAIL_PATH", exome_multi))
    if micro_mt:
        candidates.append(("MICROARRAY_HAIL_STORAGE_PATH", micro_mt))

    if not candidates:
        raise RuntimeError(
            "No candidate genotype MatrixTable env vars set "
            "(WGS_EXOME_SPLIT_HAIL_PATH / WGS_EXOME_MULTI_HAIL_PATH / MICROARRAY_HAIL_STORAGE_PATH)."
        )

    for label, path in candidates:
        print(f"Searching for rs1052553 in {label}: {path}")
        mt = hl.read_matrix_table(path)

        row_fields = set(mt.row)
        mt_snp = None

        if "rsid" in row_fields:
            mt_snp = mt.filter_rows(mt.rsid == "rs1052553")
        elif "ID" in row_fields:
            mt_snp = mt.filter_rows(mt.ID == "rs1052553")

        if mt_snp is None:
            print(f"  No 'rsid' or 'ID' field in {label}; skipping.")
            continue

        n = mt_snp.count_rows()
        if n > 0:
            print(f"  Found rs1052553 in {label} with {n} row(s).")
            return label, mt_snp

        print(f"  rs1052553 not found in {label} by rsid; trying next candidate.")

    raise RuntimeError(
        "rs1052553 not found in any candidate MatrixTable. "
        "Verify that this SNP is present and rsid annotations exist in AoU v8."
    )


def build_rs1052553_additive_cohort():
    """
    Build additive genotype-defined cohort for rs1052553 from AoU v8.

    Logic:
      - Initialize Hail in AoU RW environment.
      - Find an MT containing rs1052553 by rsID (see select_mt_with_rs1052553).
      - Compute additive dosage:
          rs1052553_dosage = number of alt alleles (0, 1, or 2)
      - Map sample IDs to person_id (int).
      - Drop missing dosages, deduplicate.
      - Write: cohort_rs1052553_additive_raw.csv

    Returns:
      Path to cohort CSV.
    """

    print("Initializing Hail for rs1052553 additive cohort...")
    hl.init(app_name="rs1052553_additive_phewas",
            quiet=True,
            log="/tmp/hail_rs1052553.log")

    label, mt_snp = select_mt_with_rs1052553()

    if "GT" not in mt_snp.entry:
        hl.stop()
        raise RuntimeError(
            f"Selected MatrixTable {label} does not have a 'GT' field; "
            "cannot compute additive dosage."
        )

    # Compute dosage per sample: number of alt alleles
    mt_snp = mt_snp.select_entries(
        rs1052553_dosage=mt_snp.GT.n_alt_alleles()
    )
    ht = mt_snp.entries().select("rs1052553_dosage")

    print("Collecting rs1052553 dosages to a local table...")
    df = ht.to_pandas()

    # Identify sample ID column
    sample_col = None
    for cand in ("s", "sample_id", "participant_id", "person_id"):
        if cand in df.columns:
            sample_col = cand
            break

    if sample_col is None:
        hl.stop()
        raise RuntimeError(
            "Cannot find sample ID column in rs1052553 entries table. "
            f"Columns: {list(df.columns)}"
        )

    df = df[[sample_col, "rs1052553_dosage"]].dropna(subset=["rs1052553_dosage"])

    # Standardize to person_id as int
    df = df.rename(columns={sample_col: "person_id"})
    df["person_id"] = df["person_id"].astype("int64")
    df["rs1052553_dosage"] = df["rs1052553_dosage"].astype("int64")
    df = df.drop_duplicates(subset=["person_id"])

    out = "cohort_rs1052553_additive_raw.csv"
    df.to_csv(out, index=False)

    print(
        f"Additive rs1052553 cohort written to {out} "
        f"(n={len(df)}) from {label}."
    )

    hl.stop()
    return out


# -------------- helpers: AoU covariates & phecode counts ----------------------


def add_aou_covariates(cohort_path):
    """
    Enrich rs1052553 cohort with AoU covariates via PheTK.Cohort.add_covariates:
      - age_at_last_event
      - sex_at_birth
      - first 10 genetic PCs
      - drop rows missing any of these

    Writes:
      cohort_rs1052553_additive_cov.csv

    Returns:
      Path to enriched cohort CSV.
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
    Use PheTK.Phecode with platform='aou' to:
      - pull ICD from All of Us CDR
      - map to phecodeX
      - aggregate counts per person/phecode

    Writes:
      phecode_counts_x_all.csv

    Returns:
      Path to phecode counts CSV.
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
    Restrict phecode_counts to the target phecode set from ferromic mapping.

    Writes:
      phecode_counts_x_ferromic_panel.csv

    Returns:
      Path to subset CSV.
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


# -------------- helpers: infer covariates & run PheWAS ------------------------


def infer_covariates_from_cohort(cohort_path):
    """
    Inspect cohort file to determine:
      - sex column name
      - covariate columns to use (age + PCs)
    Only uses columns that actually exist.
    """
    df = pl.read_csv(cohort_path, n_rows=5, infer_schema_length=1000)
    cols = set(df.columns)

    # Sex column for PheTK's sex_at_birth_col
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

    # Age-like covariates
    if "age_at_last_event" in cols:
        covariates.append("age_at_last_event")
    if "natural_age" in cols:
        covariates.append("natural_age")

    # PCs
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
    Run PheTK.PheWAS with:
      - phecode_version = X
      - independent_variable_of_interest = 'rs1052553_dosage' (0/1/2 additive)
      - covariates inferred from cohort file
      - restricted to target_phecodes from ferromic

    Writes:
      phewas_rs1052553_additive_ferromic_panel.csv
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


# -------------- main ----------------------------------------------------------


def main():
    cdr = os.getenv("WORKSPACE_CDR")
    print("Workspace CDR:", cdr)

    if not cdr:
        print(
            "ERROR: WORKSPACE_CDR not set. "
            "Run this inside an All of Us Researcher Workbench environment."
        )
        sys.exit(1)

    # 1. Build target phecode set from ferromic Phenotype labels
    target_phecodes = get_target_phecodes_from_ferromic()

    # 2. Build rs1052553 additive cohort from AoU v8 genotype data
    cohort_raw = build_rs1052553_additive_cohort()

    # 3. Add AoU covariates
    cohort_cov = add_aou_covariates(cohort_raw)

    # 4. Build phecodeX counts from AoU EHR
    phecode_counts_all = build_phecode_counts_x()

    # 5. Subset phecode counts to ferromic-derived phecodes
    phecode_counts_subset = subset_phecode_counts(
        phecode_counts_all,
        target_phecodes,
    )

    # 6. Run PheWAS with additive rs1052553 dosage
    run_phewas(
        cohort_cov,
        phecode_counts_subset,
        target_phecodes,
    )

    print("Done.")


if __name__ == "__main__":
    main()
