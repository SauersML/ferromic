#### Steps to get imputed dosages
1. Get data on AoU:
```
curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/stats/snv_list_acaf_download.py | python3
```

Prepare the data:
```
curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/imputation/prepare_data_for_infer.py | python3
```

Get the PLS regression patch in the same location where infer_dosage.py will live:
```
curl -O https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/imputation/pls_patch.py
```

Use the trained models to infer dosage:
```
curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/imputation/infer_dosage.py | python3
```

Head to the phewas/ directory to see how to run the PheWAS itself.
