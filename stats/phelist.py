import polars as pl
import sys

INPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"
OUTPUT_FILENAME = "significant_heritability_diseases.tsv"


def main():
    print(f"Loading '{INPUT_FILENAME}'...")
    try:
        df = pl.read_csv(INPUT_FILENAME, separator="\t")
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file '{INPUT_FILENAME}' not found.")
        sys.exit(1)

    # required columns
    required = [
        "phecode",
        "disease",
        "disease_category",
        "icd9_codes",
        "icd10_codes",
        "is_h2_significant_in_any_ancestry",
        "h2_overall_REML",
        "has_ukbb_heritability_data",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("FATAL ERROR: Missing columns:", ", ".join(missing))
        sys.exit(1)

    # Cast incoming columns
    df = df.with_columns([
        pl.col("is_h2_significant_in_any_ancestry").cast(pl.Int64, strict=False),
        pl.col("h2_overall_REML").cast(pl.Float64, strict=False),
        pl.col("has_ukbb_heritability_data").cast(pl.Int64, strict=False),
    ])

    has_data = pl.col("has_ukbb_heritability_data") == 1

    not_sig = pl.col("is_h2_significant_in_any_ancestry").fill_null(0) == 0
    remove_not_sig = has_data & not_sig

    remove_reml_thresh = (
        pl.col("h2_overall_REML").is_not_null()
        & (pl.col("h2_overall_REML") < 0.15)
    )

    removal_flag = (remove_not_sig | remove_reml_thresh).fill_null(False)  # | remove_no_icd

    kept_df = df.filter(~removal_flag)

    kept_df = kept_df.with_columns(
        pl.when(pl.col("h2_overall_REML").is_not_null())
          .then(pl.col("h2_overall_REML").round(4).cast(pl.Utf8))
          .otherwise(pl.lit(""))
          .alias("h2_overall_REML")
    )

    final_df = kept_df.select([
        "phecode",
        "disease",
        "disease_category",
        "icd9_codes",
        "icd10_codes",
        "h2_overall_REML",
    ])

    final_df.write_csv(OUTPUT_FILENAME, separator="\t")

    n_all = len(df)
    n_removed = int(df.filter(removal_flag).height)
    n_kept = len(final_df)
    print("-" * 50)
    print("PROCESS COMPLETE!")
    print(f"Total diseases in input : {n_all}")
    print(f"Removed (by rules)      : {n_removed}")
    print(f"Kept (written)          : {n_kept}")
    print(f"Output file             : '{OUTPUT_FILENAME}'")
    print("-" * 50)


if __name__ == "__main__":
    main()

