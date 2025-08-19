import polars as pl

# =============================================================================
# PHASE 1: SCRIPT CONFIGURATION
# =============================================================================

# Primary PheCodeX mapping file.
PHECODE_MAP_URL = "https://raw.githubusercontent.com/nhgritctran/PheTK/main/src/PheTK/phecode/phecodeX.csv"

# UK Biobank "phenocode" mappings for enrichment.
UKBB_ICD10_MAP_URL = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD10_PHECODE_MAP_20200109.txt"
UKBB_ICD9_MAP_URL = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD9_PHECODE_MAP_20200109.txt"

# UK Biobank heritability manifest for further enrichment.
H2_MANIFEST_URL = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/h2_manifest.tsv.bgz"

# Define the name for the final, comprehensive output file.
OUTPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"

# Define the delimiter for joining lists into a single string.
LIST_DELIMITER = ";"


def main():
    """
    Main function to download, merge PheCodeX, UKBB Phenocode, and
    heritability data into a single comprehensive mapping file.
    """
    print("Starting combined PheCodeX, UKBB Phenocode, and Heritability mapping process...")

    # =========================================================================
    # PHASE 2: DATA ACQUISITION AND LOADING
    # =========================================================================
    try:
        # --- 2a: Load Primary PheCodeX Data ---
        print(f"Downloading PheCodeX data from: {PHECODE_MAP_URL}")
        phecodex_df = pl.read_csv(
            PHECODE_MAP_URL,
            schema_overrides={
                "phecode": pl.Utf8, "ICD": pl.Utf8, "flag": pl.Int8,
                "phecode_string": pl.Utf8, "category_num": pl.Utf8,
                "phecode_category": pl.Utf8, "sex": pl.Utf8,
                "icd10_only": pl.Int8, "code_val": pl.Utf8
            }
        ).select("phecode", "ICD", "flag", "phecode_string", "phecode_category")

        # --- 2b: Load UK Biobank Phenocode Data ---
        print("Downloading UKBB ICD-to-Phenocode maps...")
        ukbb_icd10_df = pl.read_csv(
            UKBB_ICD10_MAP_URL, separator='\t',
            schema_overrides={'ICD10': pl.Utf8, 'phecode': pl.Utf8}
        ).select(pl.col("ICD10").alias("ICD"), pl.col("phecode").alias("ukbb_phenocode"))

        ukbb_icd9_df = pl.read_csv(
            UKBB_ICD9_MAP_URL, separator='\t',
            schema_overrides={'ICD9': pl.Utf8, 'phecode': pl.Utf8}
        ).select(pl.col("ICD9").alias("ICD"), pl.col("phecode").alias("ukbb_phenocode"))

        ukbb_lookup_df = pl.concat([ukbb_icd9_df, ukbb_icd10_df]).unique()
        print("Created a unified UKBB lookup table.")

        # --- 2c: Load and Pre-process Heritability Data ---
        print(f"Downloading heritability data from: {H2_MANIFEST_URL}")
        h2_raw_df = pl.read_csv(
            H2_MANIFEST_URL,
            separator='\t',
            null_values="NA",
            schema_overrides={
                'phenocode': pl.Utf8, 'pop': pl.Utf8,
                'estimates.final.h2_observed': pl.Float64,
                'estimates.final.h2_observed_se': pl.Float64
            }
        )

        h2_summary_lazy_df = (
            h2_raw_df.lazy()
            .select(
                pl.col("phenocode").alias("ukbb_phenocodes"),
                pl.col("pop"),
                pl.col("estimates.final.h2_observed"),
                pl.col("estimates.final.h2_observed_se")
            )
            .with_columns(
                (
                    (pl.col("estimates.final.h2_observed") - 0.01) /
                     pl.col("estimates.final.h2_observed_se") > 1.645
                ).fill_null(False).alias("is_significant")
            )
            .group_by("ukbb_phenocodes")
            .agg(
                pl.col("is_significant").any().cast(pl.UInt8).alias("is_h2_significant_in_any_ancestry"),
                pl.col("estimates.final.h2_observed").filter(pl.col("pop") == "EUR").first().alias("h2_eur")
            )
        )
        print("Successfully processed heritability data into a lazy summary table.")

    except Exception as e:
        print("---" * 20)
        print("FATAL ERROR: Failed to download or parse a required data file.")
        print(f"Error details: {e}")
        print("---" * 20)
        return

    # =========================================================================
    # PHASE 3: MERGING AND AGGREGATION
    # =========================================================================
    print("Enriching data with UKBB Phenocodes and Heritability stats...")

    base_df = phecodex_df.join(ukbb_lookup_df, on="ICD", how="left")

    grouping_cols = ["phecode", "phecode_string", "phecode_category"]
    aggregated_df = base_df.group_by(grouping_cols).agg([
        pl.col("ICD").filter(pl.col("flag") == 9).alias("ICD9_Mappings"),
        pl.col("ICD").filter(pl.col("flag") == 10).alias("ICD10_Mappings"),
        pl.col("ukbb_phenocode").drop_nulls().unique().sort().alias("ukbb_phenocodes")
    ])

    # Calculate heritability metrics only for the diseases that have phenocodes.
    h2_enriched_data = (
        aggregated_df.lazy()
        .filter(pl.col("ukbb_phenocodes").list.len() > 0)
        .explode("ukbb_phenocodes")
        .join(h2_summary_lazy_df, on="ukbb_phenocodes", how="left")
        .group_by(grouping_cols)
        .agg(
            pl.col("is_h2_significant_in_any_ancestry").max(),
            pl.col("h2_eur").mean().alias("h2_eur_avg")
        )
        .collect()
    )

    # =========================================================================
    # PHASE 4: FINAL FORMATTING AND OUTPUT GENERATION
    # =========================================================================
    print("Formatting final data and preparing for output...")
    
    # FIX: Use a single LEFT JOIN to enrich the full dataset.
    # This is more efficient and robust than the previous anti-join/vstack method.
    final_df = aggregated_df.join(h2_enriched_data, on=grouping_cols, how="left")

    # Format all columns for the final output.
    final_df = final_df.with_columns([
        pl.col("ICD9_Mappings").list.join(LIST_DELIMITER).fill_null(""),
        pl.col("ICD10_Mappings").list.join(LIST_DELIMITER).fill_null(""),
        pl.col("ukbb_phenocodes").list.join(LIST_DELIMITER).fill_null(""),
        pl.col("is_h2_significant_in_any_ancestry").fill_null(0).cast(pl.UInt8),
        pl.col("h2_eur_avg").round(4).cast(pl.Utf8).fill_null("")
    ])

    final_df = final_df.select([
        pl.col("phecode"),
        pl.col("ukbb_phenocodes").alias("ukbb_phenocode"),
        pl.col("phecode_string").alias("disease"),
        pl.col("phecode_category").alias("disease_category"),
        pl.col("is_h2_significant_in_any_ancestry"),
        pl.col("h2_eur_avg"),
        pl.col("ICD9_Mappings").alias("icd9_codes"),
        pl.col("ICD10_Mappings").alias("icd10_codes")
    ])

    try:
        final_df.write_csv(OUTPUT_FILENAME, separator='\t')
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the enriched mapping file: {OUTPUT_FILENAME}")
        print(f"Found and processed {len(final_df)} unique diseases.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
