
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forest import (
    plot_forest,
    compute_padded_or_range,
    map_inversion_series,
    load_and_prepare
)

FAMILY_FILE = "data/family_phewas.tsv"
MAIN_FILE = "data/phewas_results.tsv"
# Output files will be created in current directory then moved, or we can specify path
OUT_PDF = "family_vs_main_forest.pdf"
OUT_PNG = "family_vs_main_forest.png"

# Phenotype mapping: Family Label -> Main Label (or keyword to search)
PHENO_MAP = {
    "Breast Cancer": "Malignant_neoplasm_of_the_breast",
    "Obesity": "Obesity",
    "Heart Failure": "Heart_failure",
    "Cognitive Impairment": "Mild_cognitive_impairment"
}

def load_family_data(path):
    df = pd.read_csv(path, sep="\t")
    # Normalize columns
    # family file has: phenotype, inversion, OR, CI_low, CI_high, p
    df = df.rename(columns={
        "phenotype": "Phenotype",
        "inversion": "Inversion",
        "CI_low": "OR_lo",
        "CI_high": "OR_hi",
        "p": "Q_GLOBAL" # Using p-value as Q for plotting sizing/color if Q not available
    })
    return df

def load_main_data(path):
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Basic cleaning
    df["Phenotype"] = df["Phenotype"].fillna("").astype(str)
    df["Inversion"] = df["Inversion"].fillna("").astype(str)

    df["OR"] = pd.to_numeric(df["OR"], errors="coerce")
    df["Q_GLOBAL"] = pd.to_numeric(df["Q_GLOBAL"], errors="coerce")

    # Parse CIs
    n = len(df)
    lo = np.full(n, np.nan, dtype=float)
    hi = np.full(n, np.nan, dtype=float)

    if ("OR_Lower" in df.columns) and ("OR_Upper" in df.columns):
         lo = pd.to_numeric(df["OR_Lower"], errors="coerce").to_numpy()
         hi = pd.to_numeric(df["OR_Upper"], errors="coerce").to_numpy()

    # Re-implementing simple CI parser
    for i in df.index:
        if np.isnan(lo[i]) or np.isnan(hi[i]):
            # Try OR_CI95
            val = str(df.at[i, "OR_CI95"])
            if "," in val:
                try:
                    l, h = map(float, val.split(","))
                    lo[i] = l
                    hi[i] = h
                except:
                    pass

            # If still nan, try Wald...
            if np.isnan(lo[i]) and "Wald_OR_CI95" in df.columns:
                val = str(df.at[i, "Wald_OR_CI95"])
                if "," in val:
                    try:
                        l, h = map(float, val.split(","))
                        lo[i] = l
                        hi[i] = h
                    except:
                        pass

    df["OR_lo"] = lo
    df["OR_hi"] = hi

    return df

def main():
    print("Loading data...")
    # Check if files exist relative to CWD
    if not os.path.exists(FAMILY_FILE):
        print(f"Error: {FAMILY_FILE} not found. Please run from project root.")
        sys.exit(1)

    df_fam = load_family_data(FAMILY_FILE)
    df_main = load_main_data(MAIN_FILE)

    combined_rows = []

    for fam_label, main_label in PHENO_MAP.items():
        print(f"Processing {fam_label} vs {main_label}...")

        # Find family row
        fam_row = df_fam[df_fam["Phenotype"] == fam_label]
        if fam_row.empty:
            print(f"  Warning: Family phenotype '{fam_label}' not found.")
            continue
        fam_row = fam_row.iloc[0]

        # Find main row
        main_rows = df_main[df_main["Phenotype"] == main_label]

        # Filter by Inversion match
        fam_inv = str(fam_row["Inversion"]).lower()

        if main_rows.empty:
             print(f"  Warning: Main phenotype '{main_label}' not found.")
             match_row = None
        else:
            match_row = None
            for idx, row in main_rows.iterrows():
                main_inv = str(row["Inversion"]).lower()
                if main_inv == fam_inv:
                    match_row = row
                    break

            if match_row is None:
                print(f"  Warning: No matching inversion '{fam_inv}' found for phenotype '{main_label}' in main data.")

        section_name = fam_label

        if match_row is not None:
            combined_rows.append({
                "Inversion": section_name,
                "Phenotype": "Main PheWAS",
                "OR": float(match_row["OR"]),
                "OR_lo": float(match_row["OR_lo"]),
                "OR_hi": float(match_row["OR_hi"]),
                "Q_GLOBAL": float(match_row["Q_GLOBAL"]) if "Q_GLOBAL" in match_row else 0.05
            })

        combined_rows.append({
            "Inversion": section_name,
            "Phenotype": "Family History",
            "OR": float(fam_row["OR"]),
            "OR_lo": float(fam_row["OR_lo"]),
            "OR_hi": float(fam_row["OR_hi"]),
            "Q_GLOBAL": float(fam_row["Q_GLOBAL"])
        })

    if not combined_rows:
        print("No data to plot.")
        return

    df_plot = pd.DataFrame(combined_rows)

    or_range = compute_padded_or_range(df_plot["OR_lo"], df_plot["OR_hi"])

    print("Generating plot...")
    plot_forest(
        df_plot,
        or_range=or_range,
        out_pdf=OUT_PDF,
        out_png=OUT_PNG,
        legend_position="top_right"
    )
    print("Done.")

if __name__ == "__main__":
    main()
