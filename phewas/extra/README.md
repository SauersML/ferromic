PheWAS results, correlating ORs of the imputed dosage vs. tagging-SNP-derived dosage:
<img width="1326" height="1278" alt="image" src="https://github.com/user-attachments/assets/50d56d05-7e2b-45cb-b0c8-e051cf9fcf0a" />

To do the custom polygenic score control follow up, first install gnomon:
[https://github.com/SauersML/gnomon]([url](https://github.com/SauersML/gnomon))

Download the microarray data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```
