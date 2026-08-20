# Imputation models

The 158 trained PLS imputation models (`*.model.joblib`) and their SNP specs
(`*.snps.json`) are **not stored in this repository** — they total ~607 MB and
would bloat every clone and "Download ZIP". They are hosted as assets on the
GitHub Release:

> https://github.com/SauersML/ferromic/releases/tag/imputation-models-v1

## How they are consumed

`imputation/infer_dosage.py` and `imputation/prepare_data_for_infer.py` download only
the requested model/SNP files through the plain-text manifest:

> https://github.com/SauersML/ferromic/releases/download/imputation-models-v1/models.manifest.txt

The manifest lists one immutable release URL per `.model.joblib` and `.snps.json`.
Downloaded model files are cached in the directory passed to `infer_dosage.py` as
`--model-dir`.

## Regenerating / republishing

Models are produced by `imputation/linked.py`. To publish a new set:

```bash
gh release create imputation-models-vN --title "..." --target main
gh release upload imputation-models-vN data/models/*.model.joblib data/models/*.snps.json
# regenerate the manifest of asset URLs and upload it as models.manifest.txt
```

Then update `MODEL_MANIFEST_URL` in both inference scripts.
