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
# Phenotype column for phenotype name

# OUTPUT: raw data mapping / Gemini parsed responses. so one file with the responses for each. and one file with:
# Current phenotype label --> one or multiple other external phenotype labels (full list) AND ALSO, "best" matched phenotype

# What info to send?

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
