import polars as pl
import sys

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================

# The input file created by the previous, more complex script.
INPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"

# The name for the final, filtered output file.
OUTPUT_FILENAME = "significant_heritability_diseases.tsv"

# =============================================================================

def main():
    """
    Reads the enriched mapping file, filters it based on heritability and the
    presence of ICD codes, and outputs a six-column file with an added overall
    heritability column (h2_overall_RE) as the final column.
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

    # Sanity check: required columns must exist (schema from Program One).
    required_cols = [
        "is_h2_significant_in_any_ancestry",
        "icd9_codes",
        "icd10_codes",
        "phecode",
        "disease",
        "disease_category",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("---" * 20)
        print("FATAL ERROR: Input file is missing required columns:")
        for c in missing:
            print(f" - {c}")
        print("This script expects the exact schema produced by the previous script.")
        print("---" * 20)
        sys.exit(1)

    # --- 2. Apply the Filtering Rules ---
    # Robust cast: ensure the flag compares numerically to 1 even if parsed as string.
    heritability_filter = (
        pl.col("is_h2_significant_in_any_ancestry").cast(pl.Int64, strict=False) == 1
    )

    # Keep rows where there is at least one ICD-9 OR at least one ICD-10 code.
    # An empty string "" indicates no codes were found for that category.
    icd_code_filter = (pl.col("icd9_codes") != "") | (pl.col("icd10_codes") != "")

    # Combine both rules using an AND (&) condition.
    filtered_df = df.filter(heritability_filter & icd_code_filter)

    print(f"Filtered down to {len(filtered_df)} rows that meet the criteria.")

    # --- 3. Prepare the overall heritability display column ---
    # We expect Program One to provide a numeric 'h2_overall_RE' at the disease level.
    # If it's present, format to 4 decimals; if not, emit empty strings and warn.
    has_overall = "h2_overall_RE" in filtered_df.columns

    if has_overall:
        filtered_df = filtered_df.with_columns(
            pl.when(pl.col("h2_overall_RE").cast(pl.Float64, strict=False).is_not_null())
              .then(pl.col("h2_overall_RE").cast(pl.Float64, strict=False).round(4).cast(pl.Utf8))
              .otherwise(pl.lit(""))
              .alias("_h2_overall_RE_str")
        )
    else:
        print("WARNING: Input missing 'h2_overall_RE'; the final column will be empty.")
        filtered_df = filtered_df.with_columns(pl.lit("").alias("_h2_overall_RE_str"))

    # --- 4. Select and Order the Final Columns ---
    # Output schema: 6 columns with h2_overall_RE appended as the final column.
    final_df = filtered_df.select([
        "phecode",
        "disease",
        "disease_category",
        "icd9_codes",
        "icd10_codes",
        pl.col("_h2_overall_RE_str").alias("h2_overall_RE"),
    ])

    # --- 5. Write the Output File ---
    try:
        final_df.write_csv(OUTPUT_FILENAME, separator="\t")
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the filtered file: '{OUTPUT_FILENAME}'")
        print(f"The file contains {len(final_df)} diseases with significant heritability.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
