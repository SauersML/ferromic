# Imputation models

The 158 trained PLS imputation models (`*.model.joblib`) and their SNP specs
(`*.snps.json`) are **not stored in this repository** — they total ~607 MB and
would bloat every clone and "Download ZIP". They are hosted as assets on the
GitHub Release:

> https://github.com/SauersML/ferromic/releases/tag/imputation-models-v1

## How they are consumed

`imputation/infer_dosage.py` and `imputation/prepare_data_for_infer.py` default to
`MODEL_SOURCE = "release"` and download each required model/SNP file on demand via
the plain-text manifest:

> https://github.com/SauersML/ferromic/releases/download/imputation-models-v1/models.manifest.txt

The manifest lists one download URL per line for every `.model.joblib` and
`.snps.json`. Fallback sources remain available by setting `MODEL_SOURCE` to
`"github"` or `"s3"`, or by overriding `MODEL_MANIFEST_URL` / `MANIFEST_URL`.
If a local `data/models/` directory is populated, it is used directly (no download).

## Regenerating / republishing

Models are produced by `imputation/linked.py`. To publish a new set:

```bash
gh release create imputation-models-vN --title "..." --target main
gh release upload imputation-models-vN data/models/*.model.joblib data/models/*.snps.json
# regenerate the manifest of asset URLs and upload it as models.manifest.txt
```

Then bump the `release` URL in the two `_MANIFEST_URLS` dicts.
