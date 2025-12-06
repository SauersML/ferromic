import io
import sys

import pandas as pd
import requests

#chr7:73113989-74799029  chr7-73113990-INV-1685041
#chr16:14954790-15100859 chr16-15028481-INV-133352
#chr10:79542901-80217413 chr10-79542902-INV-674513
#chr16:28471892-28637651 chr16-28471894-INV-165758
#chr7:65219157-65531823  chr7-65219158-INV-312667
#chr17:45585159-46292045 chr17-45585160-INV-706887
#chr15:30618103-32153204 chr15-30618104-INV-1535102
#chr8:7301024-12598379   chr8-7301025-INV-5297356
#chr12:46896694-46915975 chr12-46897663-INV-16289
#chr7:54234014-54308393  chr7-54220528-INV-101153

def download_tsv(url):
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), sep="\t")


def select_q_column(df, preferred_order):
    for name in preferred_order:
        if name in df.columns:
            return name
    normalized_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", "").replace("-", "")
        normalized_map[key] = col
    for key in ("qvalue", "qval", "q"):
        if key in normalized_map:
            return normalized_map[key]
    raise KeyError("Could not identify q-value column in dataframe.")


def is_region_string(value):
    text = str(value).strip()
    if ":" not in text or "-" not in text:
        return False
    chrom_part, rest = text.split(":", 1)
    start_part, sep, end_part = rest.partition("-")
    if sep != "-":
        return False
    start_part = start_part.replace(",", "")
    end_part = end_part.replace(",", "")
    if not start_part.isdigit() or not end_part.isdigit():
        return False
    if not chrom_part:
        return False
    return True


def select_region_column(df, explicit_candidates=None):
    if explicit_candidates is None:
        explicit_candidates = []
    for name in explicit_candidates:
        if name in df.columns:
            return name
    candidate_scores = {}
    for col in df.columns:
        series = df[col].dropna().head(50)
        if series.empty:
            continue
        matches = series.map(is_region_string)
        score = matches.mean()
        candidate_scores[col] = score
    if not candidate_scores:
        return None
    best_col = max(candidate_scores, key=candidate_scores.get)
    if candidate_scores[best_col] == 0:
        return None
    return best_col


def normalize_region(region_str):
    text = str(region_str).strip()
    if not text:
        return text
    if text.startswith("chr"):
        return text
    return "chr" + text


def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main():
    cds_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/cds_conservation_table.tsv"
    snps_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/best_tagging_snps_qvalues.tsv"
    recurrence_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/balanced_recurrence_results.tsv"
    properties_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/inv_properties.tsv"

    print("--- Downloading and processing cds_conservation_table.tsv ---")
    df_cds = download_tsv(cds_url)
    q_col_cds = select_q_column(df_cds, ["q-value", "q_value"])
    cds_filtered = df_cds[df_cds[q_col_cds] < 0.05].copy()
    region_col_cds = select_region_column(cds_filtered, explicit_candidates=["Inversion", "region"])
    if region_col_cds is None:
        print("ERROR: Could not identify region-like column in cds_conservation_table.tsv")
        sys.exit(1)
    cds_regions_raw = cds_filtered[region_col_cds].astype(str).tolist()
    print(f"Found {len(cds_regions_raw)} rows with {q_col_cds} < 0.05 in cds_conservation_table.tsv")
    for r in cds_regions_raw:
        print(r)
    print()

    print("--- Downloading and processing best_tagging_snps_qvalues.tsv ---")
    df_snps = download_tsv(snps_url)
    q_col_snps = select_q_column(df_snps, ["q_value", "q-value"])
    snps_filtered = df_snps[df_snps[q_col_snps] < 0.05].copy()
    region_col_snps = select_region_column(snps_filtered, explicit_candidates=["region", "Inversion"])
    if region_col_snps is None:
        print("ERROR: Could not identify region-like column in best_tagging_snps_qvalues.tsv")
        sys.exit(1)
    snps_regions_raw = snps_filtered[region_col_snps].astype(str).tolist()
    print(f"Found {len(snps_regions_raw)} rows with {q_col_snps} < 0.05 in best_tagging_snps_qvalues.tsv")
    for r in snps_regions_raw:
        print(r)
    print()

    normalized_cds_regions = [normalize_region(r) for r in cds_regions_raw]
    normalized_snps_regions = [normalize_region(r) for r in snps_regions_raw]
    cds_region_set = set(normalized_cds_regions)
    snps_region_set = set(normalized_snps_regions)
    both_set = cds_region_set.intersection(snps_region_set)
    final_region_list = unique_preserve_order(normalized_cds_regions + normalized_snps_regions)

    print("--- FINAL LIST of regions (normalized to chr:start-end) ---")
    print("Region\tStatus")
    for region in final_region_list:
        if region in both_set:
            print(f"{region}\tPRESENT IN BOTH")
        else:
            print(f"{region}\tfrom_single_list")
    print()

    print("--- Downloading and processing balanced_recurrence_results.tsv ---")
    df_recur = download_tsv(recurrence_url)
    required_cols = ["Chromosome", "Start", "End", "Inversion_ID"]
    for col in required_cols:
        if col not in df_recur.columns:
            print(f"ERROR: Required column {col} is missing in balanced_recurrence_results.tsv")
            sys.exit(1)
    df_recur["region"] = (
        df_recur["Chromosome"].astype(str).str.strip()
        + ":"
        + df_recur["Start"].astype(int).astype(str)
        + "-"
        + df_recur["End"].astype(int).astype(str)
    )
    region_to_inversion_id = dict(zip(df_recur["region"], df_recur["Inversion_ID"]))

    final_inversion_ids = []
    print("--- Inversion_ID lookups for FINAL LIST regions ---")
    for region in final_region_list:
        inv_id = region_to_inversion_id.get(region)
        if inv_id is None:
            print(f"WARNING: No Inversion_ID found for region {region}")
        else:
            print(f"{region}\t{inv_id}")
            if inv_id not in final_inversion_ids:
                final_inversion_ids.append(inv_id)
    print()

    print("--- Downloading and processing inv_properties.tsv ---")
    df_props = download_tsv(properties_url)
    if "OrigID" not in df_props.columns:
        print("ERROR: OrigID column not found in inv_properties.tsv")
        sys.exit(1)
    matched_props = df_props[df_props["OrigID"].isin(final_inversion_ids)].copy()
    matched_ids = set(matched_props["OrigID"].astype(str).unique())
    requested_ids = set(str(x) for x in final_inversion_ids)
    missing_ids = sorted(requested_ids - matched_ids)
    if missing_ids:
        print("WARNING: The following Inversion_ID values from the shortlist were not found in inv_properties.tsv (OrigID):")
        for mid in missing_ids:
            print(mid)
        print()

    if matched_props.empty:
        print("No rows in inv_properties.tsv matched the shortlisted Inversion_ID values; cannot compute percentages.")
        return

    consensus_col = "0_single_1_recur_consensus"
    if consensus_col not in matched_props.columns:
        print(f"ERROR: Column {consensus_col} not found in inv_properties.tsv")
        sys.exit(1)

    consensus_series = matched_props[consensus_col]
    numeric = pd.to_numeric(consensus_series, errors="coerce")
    is_one = numeric == 1
    is_zero = numeric == 0
    is_other = ~(is_one | is_zero)

    total = float(len(consensus_series))
    pct_one = is_one.sum() * 100.0 / total
    pct_zero = is_zero.sum() * 100.0 / total
    pct_other = is_other.sum() * 100.0 / total

    print("--- 0_single_1_recur_consensus value percentages among matched inversions ---")
    print(f"Value 1: {pct_one:.2f}%")
    print(f"Value 0: {pct_zero:.2f}%")
    print(f"Other values: {pct_other:.2f}%")
    print()

    # New analysis: Imputation-based filtering
    print("--- Downloading and processing imputation_results_merged.tsv ---")
    imputation_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/imputation_results_merged.tsv"
    df_imputation = download_tsv(imputation_url)
    
    # Identify the ID column in imputation data
    id_col_candidates = ["Inversion_ID", "OrigID", "inversion_id", "ID"]
    imputation_id_col = None
    for candidate in id_col_candidates:
        if candidate in df_imputation.columns:
            imputation_id_col = candidate
            break
    
    if imputation_id_col is None:
        print("ERROR: Could not find ID column in imputation_results_merged.tsv")
        print(f"Available columns: {list(df_imputation.columns)}")
        sys.exit(1)
    
    # Check for required columns
    required_imputation_cols = ["unbiased_pearson_r2", "p_fdr_bh"]
    for col in required_imputation_cols:
        if col not in df_imputation.columns:
            print(f"ERROR: Required column {col} not found in imputation_results_merged.tsv")
            sys.exit(1)
    
    # Merge inv_properties with imputation results
    merged_df = df_props.merge(df_imputation, left_on="OrigID", right_on=imputation_id_col, how="inner")
    
    # After merge, column names may have suffixes. Use the version from df_props (_x suffix)
    consensus_col_merged = "0_single_1_recur_consensus_x" if "0_single_1_recur_consensus_x" in merged_df.columns else "0_single_1_recur_consensus"
    
    # Filter for List One criteria
    list_one_df = merged_df[
        (merged_df[consensus_col_merged].isin([0, 1])) &
        (merged_df["unbiased_pearson_r2"] > 0.5) &
        (merged_df["p_fdr_bh"] < 0.05)
    ].copy()
    
    # Use OrigID from the merged dataframe (may have suffix)
    origid_col = "OrigID_x" if "OrigID_x" in merged_df.columns else "OrigID"
    list_one_ids = list_one_df[origid_col].tolist()
    
    print("--- LIST ONE: Imputation-filtered inversions ---")
    print(f"Criteria: 0_single_1_recur_consensus in [0,1], unbiased_pearson_r2 > 0.5, p_fdr_bh < 0.05")
    print(f"Total inversions meeting criteria: {len(list_one_ids)}")
    print()
    for inv_id in list_one_ids:
        print(inv_id)
    print()
    
    # Create List Two (intersection with final shortlist)
    list_one_set = set(list_one_ids)
    final_shortlist_set = set(final_inversion_ids)
    list_two_ids = [inv_id for inv_id in list_one_ids if inv_id in final_shortlist_set]
    
    print("--- LIST TWO: Imputation-filtered inversions ALSO in final shortlist ---")
    print(f"Criteria: Meet List One criteria AND q-value < 0.05 in CDS conservation OR SNP tagging")
    print(f"Total inversions: {len(list_two_ids)}")
    print()
    for inv_id in list_two_ids:
        print(inv_id)
    print()


if __name__ == "__main__":
    main()
