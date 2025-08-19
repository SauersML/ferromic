import polars as pl

# =============================================================================
# PHASE 1: SCRIPT CONFIGURATION
# =============================================================================

# The direct URL to the raw PheCodeX mapping file on GitHub.
PHECODE_MAP_URL = "https://raw.githubusercontent.com/nhgritctran/PheTK/main/src/PheTK/phecode/phecodeX.csv"

# Define the name for the final output file
OUTPUT_FILENAME = "phecodex_us_icd_mappings.tsv"

# Define the delimiter for joining lists of ICD codes into a single string.
LIST_DELIMITER = ";"


def main():
    """
    Main function to execute the entire data download, extraction, and
    transformation pipeline.
    """
    print("Starting PheCodeX mapping extraction process...")

    # =========================================================================
    # PHASE 2: DATA ACQUISITION AND LOADING
    # =========================================================================
    print(f"Downloading mapping data from: {PHECODE_MAP_URL}")

    try:
        # Define the complete schema for the CSV file to prevent incorrect type inference.
        # This is the key fix for the parsing error.
        csv_schema = {
            "phecode": pl.Utf8,
            "ICD": pl.Utf8,
            "flag": pl.Int8,
            "phecode_string": pl.Utf8,
            "category_num": pl.Utf8,       # Read as string for safety
            "phecode_category": pl.Utf8,
            "sex": pl.Utf8,
            "icd10_only": pl.Int8,
            "code_val": pl.Utf8            # Read as string to handle values like '002.1'
        }

        # Polars can read directly from a URL. We use `schema_overrides` (the modern
        # argument) to enforce the correct data types for all columns.
        mapping_df = pl.read_csv(
            PHECODE_MAP_URL,
            schema_overrides=csv_schema
        )
        print(f"Successfully downloaded and loaded {len(mapping_df)} mappings.")

    except Exception as e:
        # This will catch network errors (no internet, 404) or parsing errors.
        print("---" * 20)
        print("FATAL ERROR: Failed to download or parse the data file.")
        print(f"Please check your internet connection and that the URL is still valid:")
        print(f"URL: {PHECODE_MAP_URL}")
        print(f"Error details: {e}")
        print("---" * 20)
        return  # Exit the script

    # =========================================================================
    # PHASE 3: CORE DATA TRANSFORMATION AND AGGREGATION
    # =========================================================================
    print("Transforming and aggregating data...")

    # 1. Separate the DataFrame into ICD-9 and ICD-10 based on the 'flag' column.
    icd9_df = mapping_df.filter(pl.col("flag") == 9)
    icd10_df = mapping_df.filter(pl.col("flag") == 10)

    # 2. Define the columns that uniquely identify a disease.
    grouping_cols = ["phecode", "phecode_string", "phecode_category"]

    # 3. Aggregate ICD-9 codes into a list for each unique disease.
    icd9_agg_df = icd9_df.group_by(grouping_cols).agg(
        pl.col("ICD").alias("ICD9_Mappings")
    )

    # 4. Aggregate ICD-10 codes into a list for each unique disease.
    icd10_agg_df = icd10_df.group_by(grouping_cols).agg(
        pl.col("ICD").alias("ICD10_Mappings")
    )

    # 5. Combine the aggregated data using an 'outer' join.
    # This is crucial to retain diseases that may only have ICD-9 or ICD-10 maps.
    merged_df = icd9_agg_df.join(
        icd10_agg_df, on=grouping_cols, how="outer"
    )

    # =========================================================================
    # PHASE 4: FINAL FORMATTING AND OUTPUT GENERATION
    # =========================================================================
    print("Formatting final data and preparing for output...")

    # 1. Handle potential nulls from the outer join by replacing them with empty lists,
    #    then convert the list columns into delimited strings for the TSV output.
    final_df = merged_df.with_columns([
        pl.when(pl.col("ICD9_Mappings").is_null())
          .then(pl.lit(""))  # Use empty string for null lists
          .otherwise(pl.col("ICD9_Mappings").list.join(LIST_DELIMITER))
          .alias("ICD9_Mappings"),
        pl.when(pl.col("ICD10_Mappings").is_null())
          .then(pl.lit(""))  # Use empty string for null lists
          .otherwise(pl.col("ICD10_Mappings").list.join(LIST_DELIMITER))
          .alias("ICD10_Mappings")
    ])

    # 2. Select and rename columns to the final desired format.
    # The order of columns in .select() determines the final order in the TSV.
    final_df = final_df.select([
        pl.col("phecode"),
        pl.col("phecode_string").alias("disease"),
        pl.col("phecode_category").alias("disease_category"),
        pl.col("ICD9_Mappings").alias("icd9_codes"),
        pl.col("ICD10_Mappings").alias("icd10_codes")
    ])

    # 3. Write the final, clean DataFrame to a TSV file.
    try:
        final_df.write_csv(OUTPUT_FILENAME, separator='\t')
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the mapping file: {OUTPUT_FILENAME}")
        print(f"Found and processed {len(final_df)} unique diseases.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
