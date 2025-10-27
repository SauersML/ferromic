#### Steps to get imputed dosages
1. Get data on AoU:
```
curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/stats/snv_list_acaf_download.py | python3
```

curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/stats/prepare_data_for_infer.py | python3



curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/stats/infer_dosage.py | python3
