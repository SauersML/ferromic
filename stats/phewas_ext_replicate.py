# For EACH inversion in our list
  #   For EACH significant-or-close-to-it phenotype (q-value less than 0.1) --> parallel queries
    #   Send phenotype name + ICD codes that constructed it? + context on UKBB vs. All of Us
    #   Send info about phenotype? --> randomize phenotype order to avoid first/last saliency bias
    #   Send list of TARGET phenotypes from external phewas that we want to match to
    #   List all that are identical-ish / synonymous. Also, choose BEST single matching phenotype.
    #   Recieve and save list
    #   Repeat 2-3x? for consistency /reliability
    #   specify format in prompt that it should be output in, like csv, json, etc., whatever, for easy parsing
    #   Get OR, 95% CI, p-value, phenotype name, of what we matched to in external phewas

# TOPMED phenotype + category list: data/phewas v4 - PheWeb TOPMED phenos.tsv
# https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas%20v4%20-%20PheWeb%20TOPMED%20phenos.tsv
# Format:
# phenostring	category
# Disorders of iron metabolism	hematopoietic
# etc.

# Our present phewas: https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phewas_results.tsv
# Q_GLOBAL column for q-value
# Phenotype column for phenotype name, but with underscores (so make sure you change underscores to spaces to match with other stuff)

# OUTPUT: raw data mapping / Gemini parsed responses. so one file with the responses for each. and one file with:
# Current phenotype label --> one or multiple other external phenotype labels (full list) AND ALSO, "best" matched phenotype

# What info to send?

# EXTRA info which is useful to send for OUR phenotypes can be found in https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/significant_heritability_diseases.tsv
# it has this format: 

# phecode	disease	disease_category	icd9_codes	icd10_codes	h2_overall_REML
# BI_160	Deficiency anemias	Blood/Immune	280;280.0;280.1;280.8;280.9;281;281.0;281.1;281.2;281.3;281.4;281.8;281.9	D50;D50.0;D50.1;D50.8;D50.9;D51;D51.0;D51.1;D51.2;D51.3;D51.8;D51.9;D52;D52.0;D52.1;D52.8;D52.9;D53;D53.0;D53.1;D53.2;D53.8;D53.9	0.5425
# BI_160.1	Iron deficiency anemia	Blood/Immune	280;280.0;280.1;280.8;280.9	D50;D50.0;D50.1;D50.8;D50.9	0.5011
# BI_160.11	Iron deficiency anemia secondary to blood loss	Blood/Immune	280.0	D50.0	0.5011
# BI_160.2	Megaloblastic anemia	Blood/Immune	281.0;281.1;281.2;281.3	D51;D51.0;D51.1;D51.2;D51.3;D51.8;D51.9;D52;D52.0;D52.1;D52.8;D52.9;D53.1	1.8703

# We can send 1. disease name, 2. disease category, 3. list of ICD-9 codes, 4. List of ICD-10 codes. This will make it more specific.

# example of how to query Gemini 3 Pro:
# pip install -U google-genai
# use env var GEMINI_API_KEY
# Client will automatically use GEMINI_API_KEY from the environment
# client = genai.Client()
# response = client.models.generate_content(
#     model="gemini-3-pro-preview",
#     contents="Say hi in one sentence and tell me a Python tip.",
# )
# print(response.text)


# https://www.nature.com/articles/s41586-021-03855-y ?
# 

#  Correlate / overlap present results. Correlate: -log(raw p), OR. Overlap: CIs. (later)
