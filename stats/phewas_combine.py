# For EACH inversion in our list
  #   For EACH significant-or-close-to-it phenotype (q-value less than 0.1) --> parallel queries
    #   Send phenotype name
    #   Send info about phenotype?
    #   Send list of TARGET phenotypes from external phewas that we want to match to
    #   Recieve and save list
    #   Repeat 2-3x? for consistency /reliability
    #   Get OR, 95% CI, p-value, phenotype name, of what we matched to in external phewas
#  Correlate / overlap present results. Correlate: -log(raw p), OR. Overlap: CIs.

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
