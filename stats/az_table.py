import pandas as pd
import requests
import io
import sys

def download_tsv(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.BytesIO(response.content), sep='\t')
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)

def main():
    # ---------------------------------------------------------
    # 1. DEFINE DATA SOURCES
    # ---------------------------------------------------------
    aou_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas%20v4%20-%20specifics%20AoU.tsv"
    
    # Specific URLs for specific chromosomes
    url_chr17 = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas%20v4%20-%20AstraZeneca%2017-45996523-A-G.tsv"
    url_chr8  = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas%20v4%20-%20AstraZeneca%208-10611723-A-G%20.tsv"

    # ---------------------------------------------------------
    # 2. DEFINE THE HARDCODED MAPPING
    # ---------------------------------------------------------
    
    list_hypo = [
        "Hypothyroidism",
        "Union#E03#Other hypothyroidism",
        "Union#E039#Hypothyroidism unspecified",
        "130697#Source of report of E03 (Other hypothyroidism)",
        "20002#1226#hypothyroidism|myxoedema"
    ]
    
    list_goiter = [
        "130699#Source of report of E04 (Other nontoxic goitre)",
        "Union#E04#Other nontoxic goitre",
        "Union#E040#Nontoxic diffuse goitre",
        "Union#E042#Nontoxic multinodular goitre"
    ]
    
    list_nevi = [
        "Union#D22#Melanocytic naevi",
        "Union#D225#Melanocytic naevi of trunk"
    ]
    
    list_rosacea = [
        "131793#Source of report of L71 (Rosacea)",
        "Union#L71#Rosacea"
    ]
    
    list_breast_cancer = [
        "20001#1002#breast cancer",
        "20002#3#gynaecology|breast",
        "40006#Block C50-C50#C50-C50 Malignant neoplasm of breast",
        "40006#C50#Malignant neoplasm of breast",
        "40006#C509#Malignant neoplasm: Breast unspecified",
        "40006#D05#Carcinoma in situ of breast",
        "40006#D051#Intraductal carcinoma in situ",
        "41202#Block C50-C50#C50-C50 Malignant neoplasm of breast",
        "41202#C50#Malignant neoplasm of breast",
        "41202#C509#Malignant neoplasm: Breast unspecified",
        "41202#D05#Carcinoma in situ of breast",
        "41202#D051#Intraductal carcinoma in situ",
        "Union#Block C50-C50#C50-C50 Malignant neoplasm of breast",
        "Union#C50#Malignant neoplasm of breast",
        "Union#C509#Malignant neoplasm: Breast unspecified",
        "Union#D05#Carcinoma in situ of breast",
        "Union#D051#Intraductal carcinoma in situ",
        "Union#D059#Carcinoma in situ of breast unspecified",
        "Union#Z853#Personal history of malignant neoplasm of breast",
        # Added based on recent review for completeness on benign neoplasms
        "Union#D24#Benign neoplasm of breast"
    ]
    
    list_myopia = [
        "41202#H527#Disorder of refraction unspecified"
    ]
    
    list_diverticular = [
        "20002#1458#diverticular disease|diverticulitis"
    ]
    
    list_dementia = [
        "130841#Source of report of F02 (Dementia in other diseases classified elsewhere)",
        "40001#G301#Alzheimer disease with late onset",
        "Union#F02#Dementia in other diseases classified elsewhere",
        "Union#F028#Dementia in other specified diseases classified elsewhere"
    ]
    
    list_obesity = [
        "130793#Source of report of E66 (Obesity)",
        "Union#Block E65-E68#E65-E68 Obesity and other hyperalimentation",
        "Union#E66#Obesity",
        "Union#E668#Other obesity",
        "Union#E669#Obesity unspecified"
    ]
    
    list_dysphagia = [
        "Union#R13#Dysphagia",
        "20002#1134#oesophageal disorder",
        "Union#K22#Other diseases of oesophagus"
    ]
    
    list_osteoporosis = [
        "Union#M819#Osteoporosis unspecified",
        "20002#1309#osteoporosis",
        "131965#Source of report of M81 (Osteoporosis without pathological fracture)"
    ]
    
    list_stroke = [
        "20002#1082#transient ischaemic attack (tia)",
        "41202#I67#Other cerebrovascular diseases",
        "Union#F011#Multi-infarct dementia",
        "Union#I634#Cerebral infarction due to embolism of cerebral arteries",
        "Union#G450#Vertebro-basilar artery syndrome"
    ]

    mapping = {
        "Hypothyroidism_not_specified_as_secondary": list_hypo,
        "Multinodular_goiter": list_goiter,
        "Goiter": list_goiter,
        "Uninodular_goiter_single_thyroid_nodule": list_goiter,
        "Melanocytic_nevi": list_nevi,
        "Rosacea": list_rosacea,
        "Malignant_neoplasm_of_the_breast_female": list_breast_cancer,
        "Malignant_neoplasm_of_the_breast": list_breast_cancer,
        "Myopia": list_myopia,
        "Diverticula_of_colon": list_diverticular,
        "Mild_cognitive_impairment": list_dementia,
        "Morbid_obesity": list_obesity,
        
        "Aphagia_and_dysphagia": list_dysphagia,
        "Osteoporosis": list_osteoporosis,
        "Cerebral_infarction_Ischemic_stroke": list_stroke
    }

    # ---------------------------------------------------------
    # 3. LOAD UKBB DATA INTO SEPARATE DICTIONARIES
    # ---------------------------------------------------------
    # Helper to load a DF into a dictionary Key->{Stats}
    def load_ukbb_dict(url):
        df = download_tsv(url)
        df.columns = df.columns.str.strip()
        data_dict = {}
        for _, row in df.iterrows():
            pheno = row['Phenotype']
            data_dict[pheno] = {
                'p': row['p-value'],
                'or': row['Odds ratio']
            }
        return data_dict

    print("Downloading Chr17 data...")
    dict_chr17 = load_ukbb_dict(url_chr17)
    
    print("Downloading Chr8 data...")
    dict_chr8 = load_ukbb_dict(url_chr8)

    # ---------------------------------------------------------
    # 4. PROCESS AoU DATA
    # ---------------------------------------------------------
    print("Downloading AoU data...")
    aou_df = download_tsv(aou_url)
    aou_df.columns = aou_df.columns.str.strip()
    
    processed_rows = []
    max_matches_found = 0
    
    for _, row in aou_df.iterrows():
        aou_pheno = row['Phenotype']
        inversion_str = str(row['Inversion']) # ensure string
        
        # Base Row Data
        row_data = {
            'Phenotype_AoU': aou_pheno,
            'BH_P_AoU': row.get('Q_GLOBAL', ''),
            'OR__AoU': row.get('OR', ''),
            'Inversion': inversion_str,
            'P_Value_AoU': row.get('P_Value_x', '')
        }
        
        # DETERMINE WHICH SHEET TO USE
        target_dict = None
        
        # STRICT LOGIC: IF Chr17 -> Use Dict17. IF Chr8 -> Use Dict8. ELSE -> None.
        if "chr17" in inversion_str:
            target_dict = dict_chr17
        elif "chr8" in inversion_str:
            target_dict = dict_chr8
        else:
            target_dict = None
        
        matches_data = []
        
        # ONLY look for matches if we have a valid target dictionary
        if target_dict is not None and aou_pheno in mapping:
            potential_matches = mapping[aou_pheno]
            
            for ukbb_str in potential_matches:
                # CHECK ONLY IN THE TARGET DICTIONARY
                if ukbb_str in target_dict:
                    stats = target_dict[ukbb_str]
                    matches_data.append({
                        'pheno': ukbb_str,
                        'p': stats['p'],
                        'or': stats['or']
                    })
        
        # SORT matches by p-value (lowest first)
        def sort_key(x):
            try:
                return float(x['p'])
            except:
                return 1.0
        
        matches_data.sort(key=sort_key)
        
        if len(matches_data) > max_matches_found:
            max_matches_found = len(matches_data)
            
        row_data['matches'] = matches_data
        processed_rows.append(row_data)
        
    # ---------------------------------------------------------
    # 5. CONSTRUCT FINAL DATAFRAME
    # ---------------------------------------------------------
    final_cols = ['Phenotype_AoU', 'BH_P_AoU', 'OR__AoU', 'Inversion', 'P_Value_AoU']
    
    for i in range(1, max_matches_found + 1):
        final_cols.append(f'Phenotype_UKBB_{i}')
        final_cols.append(f'p-value_UKBB_{i}')
        final_cols.append(f'OR_UKBB_{i}')
        
    final_rows = []
    
    for item in processed_rows:
        new_row = {
            'Phenotype_AoU': item['Phenotype_AoU'],
            'BH_P_AoU': item['BH_P_AoU'],
            'OR__AoU': item['OR__AoU'],
            'Inversion': item['Inversion'],
            'P_Value_AoU': item['P_Value_AoU']
        }
        
        matches = item['matches']
        
        for i in range(1, max_matches_found + 1):
            if i <= len(matches):
                m = matches[i-1]
                new_row[f'Phenotype_UKBB_{i}'] = m['pheno']
                new_row[f'p-value_UKBB_{i}'] = m['p']
                new_row[f'OR_UKBB_{i}'] = m['or']
            else:
                new_row[f'Phenotype_UKBB_{i}'] = ""
                new_row[f'p-value_UKBB_{i}'] = ""
                new_row[f'OR_UKBB_{i}'] = ""
                
        final_rows.append(new_row)
        
    df_out = pd.DataFrame(final_rows, columns=final_cols)
    
    output_filename = "combined_phenotypes_strict_.tsv"
    df_out.to_csv(output_filename, sep='\t', index=False)
    print(f"Successfully created {output_filename}")

if __name__ == "__main__":
    main()
