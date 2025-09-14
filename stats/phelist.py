import polars as pl
import sys

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# The input file created by Program One.
INPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"

# The name for the final, filtered output file.
OUTPUT_FILENAME = "significant_heritability_diseases.tsv"

# =============================================================================

def main():
    """
    Reads the enriched mapping file, applies three filters:
      1) any-pop significant flag == 1,
      2) at least one ICD-9 or ICD-10 code present,
      3) NON-NEGATIVE overall h^2 estimate (h2_overall_REML >= 0),
    and writes a 6-column TSV that includes the formatted overall estimate.
    """
    print(f"Starting the filtering process for '{INPUT_FILENAME}'...")

    # --- 1. Load the Data ---
    try:
        df = pl.read_csv(INPUT_FILENAME, separator="\t")
        print(f"Successfully loaded {len(df)} rows from the input file.")
    except FileNotFoundError:
        print("---" * 20)
        print(f"FATAL ERROR: The input file '{INPUT_FILENAME}' was not found.")
        print("Please make sure this script is in the same directory as the input file,")
        print("or provide the full path to the file.")
        print("---" * 20)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        sys.exit(1)

    # --- 1b. Sanity check: required columns must exist (from Program One) ---
    required_cols = [
        "is_h2_significant_in_any_ancestry",
        "icd9_codes",
        "icd10_codes",
        "phecode",
        "disease",
        "disease_category",
        "h2_overall_REML",  # final overall estimate column
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("---" * 20)
        print("FATAL ERROR: Input file is missing required columns:")
        for c in missing:
            print(f" - {c}")
        print("This script expects the schema produced by the previous script.")
        print("---" * 20)
        sys.exit(1)

    # --- 2. Apply the Filtering Rules ---
    sig_filter = pl.col("is_h2_significant_in_any_ancestry").cast(pl.Int64, strict=False) == 1
    icd_filter = (pl.col("icd9_codes") != "") | (pl.col("icd10_codes") != "")
    nonneg_overall_filter = (
        pl.col("h2_overall_REML").cast(pl.Float64, strict=False).is_not_null()
        & (pl.col("h2_overall_REML").cast(pl.Float64, strict=False) >= 0.0)
    )

    filtered_df = df.filter(sig_filter & icd_filter & nonneg_overall_filter)
    print(f"Filtered down to {len(filtered_df)} rows that meet the criteria.")

    # --- 3. Format the overall estimate to 4 decimals as a string ---
    filtered_df = filtered_df.with_columns(
        pl.col("h2_overall_REML").cast(pl.Float64, strict=False).round(4).cast(pl.Utf8)
    )

    # --- 4. Select and Order the Final Columns (6 columns) ---
    final_df = filtered_df.select([
        "phecode",
        "disease",
        "disease_category",
        "icd9_codes",
        "icd10_codes",
        "h2_overall_REML",  # formatted string (4 d.p.)
    ])

    # --- 5. Write the Output File ---
    try:
        final_df.write_csv(OUTPUT_FILENAME, separator="\t")
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the filtered file: '{OUTPUT_FILENAME}'")
        print(f"The file contains {len(final_df)} diseases passing all filters.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
